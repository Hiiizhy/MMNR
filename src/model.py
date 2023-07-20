import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config, numItems):
        super(Model, self).__init__()

        self.config = config
        self.dim = self.config.dim
        self.asp = self.config.asp

        self.ctx = self.config.ctx
        self.h1 = self.config.h1
        self.h2 = self.config.h2

        self.itemEmb = nn.Embedding(numItems + 1, self.dim, padding_idx=config.padIdx)

        # Aspect-Specific Projection Matrices (K different aspects)
        self.aspProj = nn.Parameter(torch.Tensor(self.asp, self.dim, self.h1), requires_grad=True)
        self.aspProjSeq = nn.Parameter(torch.Tensor(self.asp * 2, self.h1, self.h2), requires_grad=True)
        torch.nn.init.xavier_normal_(self.aspProj.data, gain=1)
        torch.nn.init.xavier_normal_(self.aspProjSeq.data, gain=1)

        self.out = nn.Linear(self.h2, numItems)
        self.his_linear_embds = nn.Linear(numItems, self.h2)
        self.his_nn_embds = nn.Embedding(numItems + 1, self.h2, padding_idx=config.padIdx)
        self.gate_his = nn.Linear(self.h2, 1)

        self.asp_h1_h2 = nn.Linear(self.h1, self.h2)

    def forward(self, seq, decay, uHis, iHis, device):

        batch = seq.shape[0]  # batch
        self.max_seq = seq.shape[1]  # L
        self.max_bas = seq.shape[2]  # B

        # Multi-view Embedding
        uEmbs, iEmbs = self.EmbeddingLayer(batch, seq, uHis, iHis, device)  # [batch, L, B, d]

        # Multi-aspect Representation Learning
        uEmbsAsp = self.AspectLearning(uEmbs, batch, device)  # [batch, asp, L, h1]
        iEmbsAsp = self.AspectLearning(iEmbs, batch, device)

        # decay [batch, L, 1]
        decay = decay.unsqueeze(1)  # [batch, 1, L, 1]
        decay = decay.repeat(1, self.config.asp, 1, 1)  # [batch, asp, L, 1]
        uEmbsAspDec = uEmbsAsp * decay  # decay[batch, asp, L, 1]->[batch, asp, L, h1]
        iEmbsAspDec = iEmbsAsp * decay  # decay[batch, asp, L, 1]->[batch, asp, L, h1]

        uAsp = self.asp_h1_h2(torch.sum(uEmbsAspDec, dim=2) / self.max_seq)
        iAsp = self.asp_h1_h2(torch.sum(iEmbsAspDec, dim=2) / self.max_seq)

        result, loss_cl = self.PredictionLayer(uAsp, iAsp, uHis)

        return result, loss_cl

    def EmbeddingLayer(self, batch, seq, uHis, iHis, device):
        '''
        input:
            seq [batch, L, B, d]
        output:
            userEmbs [batch, L, B, d]
            itemEmbs [batch, L, B, d]
        '''
        embs = self.itemEmb(seq)  # nn.Embedding

        # [batch*max_num_seq*max_bas]
        row = torch.arange(batch).repeat(self.max_seq * self.max_bas, 1).transpose(0, 1).reshape(-1)
        col = seq.reshape(len(seq), -1).reshape(-1)  # [batch, L, B]

        # padded = torch.zeros(batch, 1).to(device)  # [batch, 1]
        padded = torch.zeros(batch, 1).fill_(0).to(device)  # [batch, 1]
        userHis = torch.cat((uHis, padded), dim=1)  # [batch, numItems+1]
        itemHis = torch.cat((iHis, padded), dim=1)  # [batch, numItems+1]

        uMatrix = userHis[row, col].reshape(batch, self.max_seq, -1, 1)  # [batch, L, B, 1]
        iMatrix = itemHis[row, col].reshape(batch, self.max_seq, -1, 1)  # [batch, L, B, 1]

        uEmbs = embs * uMatrix
        iEmbs = embs * iMatrix

        return uEmbs, iEmbs

    def AspectLearning(self, embs, batch, device):
        '''
        input:
            uEmbs [batch, L, B, d]
            iEmbs [batch, L, B, d]
        output:
            basketAsp  [batch, asp, L, h1]
        '''

        # Aspect Embeddings (basket)
        self.aspEmbed = nn.Embedding(self.asp, self.ctx * self.h1).to(device)
        self.aspEmbed.weight.requires_grad = True
        torch.nn.init.xavier_normal_(self.aspEmbed.weight.data, gain=1)

        # Loop over all aspects
        asp_lst = []
        for a in range(self.asp):
            self.norm = nn.LayerNorm(self.aspProj[a].shape[1]).to(device)

            # [batch, L, B, d] × [d, h1] = [batch, L, B, h1]
            aspProj = torch.tanh(torch.matmul(embs, self.norm(self.aspProj[a])))

            # [batch, L, 1] -> [batch, L, 1, h1]
            aspEmbed = self.aspEmbed(torch.LongTensor(batch, self.max_seq, 1).fill_(a).to(device))
            aspEmbed = torch.transpose(aspEmbed, 2, 3)  # [batch, L, h1, 1]

            if (self.ctx == 1):
                # [batch, L, B, (1*h1)] × [batch, L, (1*h1), 1] = [batch, L, B, 1]
                aspAttn = torch.matmul(aspProj, aspEmbed)
                aspAttn = F.softmax(aspAttn, dim=2)  # [batch,L,B,1]
            else:
                pad_size = int((self.ctx - 1) / 2)

                # [batch, max_len, max_bas+1+1, h1]; pad_size=1
                aspProj_padded = F.pad(aspProj, (0, 0, pad_size, pad_size), "constant", 0)

                # [batch,L,B+1+1,h1]->[batch,L,B,h1,ctx]
                aspProj_padded = aspProj_padded.unfold(2, self.ctx, 1)  # sliding
                aspProj_padded = torch.transpose(aspProj_padded, 3, 4)
                # [batch, max_len, max_bas, ctx*h1]
                aspProj_padded = aspProj_padded.contiguous().view(-1, self.max_seq,
                                                                  self.max_bas, self.ctx * self.h1)

                # Calculate Attention: Inner Product & Softmax
                # [batch, L,B, (ctx*h1)] x [batch, L, (ctx*h1), 1] -> [batch, L, B, 1]
                aspAttn = torch.matmul(aspProj_padded, aspEmbed)
                aspAttn = F.softmax(aspAttn, dim=2)  # [batch, max_len, max_bas, 1]

            # [batch, L, B, h1] x [batch, L, B, 1]
            aspItem = aspProj * aspAttn.expand_as(aspProj)  # [batch, L, B, h1]
            batch_asp = torch.sum(aspItem, dim=2)  # [batch, L, h1]

            # [batch, L, h1] -> [batch, 1, L, h1]
            asp_lst.append(torch.unsqueeze(batch_asp, 1))

        # [batch, asp, L, h1]
        basketAsp = torch.cat(asp_lst, dim=1)

        return basketAsp

    def PredictionLayer(self, uuAsp, iiAsp, his):
        intent = []
        loss_cl = 0
        # Over loop each aspect
        for b in range(uuAsp.shape[1]):
            uInterest = torch.tanh(uuAsp[:, b, :])  # [batch, h2]
            iInterest = torch.tanh(iiAsp[:, b, :])  # [batch, h2]

            uLoss = self.cl_loss(uInterest, iInterest)  # [batch, h2]
            iLoss = self.cl_loss(iInterest, uInterest)  # [batch, h2]
            cLoss = uLoss + iLoss

            Interest = torch.cat([uInterest.unsqueeze(2), iInterest.unsqueeze(2)], dim=2)  # [batch,h2,2]
            Interests = torch.sum(Interest, dim=2)  # [batch,h2]
            scores_trans = self.out(Interests)  # [batch,h2] -> [batch,numItems]
            scores_trans = F.softmax(scores_trans, dim=-1)  # [batch, numItems]

            hisEmb = self.his_linear_embds(his)  # [batch,numItems] -> [batch,h2]

            # [h1 -> 1]
            gate = torch.sigmoid(self.gate_his(hisEmb) + self.gate_his(Interests))  # value

            res = gate * scores_trans + (1 - gate) * his  # [batch, numItems]
            res = res / math.sqrt(self.dim)

            intent.append(res.unsqueeze(2))
            loss_cl += cLoss.mean()

        results = torch.cat(intent, dim=2)  # [batch, numItems, asp]
        result = F.max_pool1d(results, int(results.size(2))).squeeze(2)  # [batch, numItems]
        loss_cl = loss_cl / self.asp

        return result, loss_cl

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def cl_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        tau = 0.6
        f = lambda x: torch.exp(x / tau)

        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
