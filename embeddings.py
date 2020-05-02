from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn


def md_solver(n, alpha, d0=None, B=None, round_dim=True, k=None):
    '''
    An external facing function call for mixed-dimension assignment
    with the alpha power temperature heuristic
    Inputs:
    n -- (torch.LongTensor) ; Vector of num of rows for each embedding matrix
    alpha -- (torch.FloatTensor); Scalar, non-negative, controls dim. skew
    d0 -- (torch.FloatTensor); Scalar, baseline embedding dimension
    B -- (torch.FloatTensor); Scalar, parameter budget for embedding layer
    round_dim -- (bool); flag for rounding dims to nearest pow of 2
    k -- (torch.LongTensor) ; Vector of average number of queries per inference
    '''
    n, indices = torch.sort(n)
    k = k[indices] if k is not None else torch.ones(len(n))
    d = alpha_power_rule(n.type(torch.float) / k, alpha, d0=d0, B=B)
    if round_dim:
        d = pow_2_round(d)
    return d


def alpha_power_rule(n, alpha, d0=None, B=None):
    if d0 is not None:
        lamb = d0 * (n[0].type(torch.float) ** alpha)
    elif B is not None:
        lamb = B / torch.sum(n.type(torch.float) ** (1 - alpha))
    else:
        raise ValueError("Must specify either d0 or B")
    d = torch.ones(len(n)) * lamb * (n.type(torch.float) ** (-alpha))
    for i in range(len(d)):
        if i == 0 and d0 is not None:
            d[i] = d0
        else:
            d[i] = 1 if d[i] < 1 else d[i]
    return (torch.round(d).type(torch.long))


def pow_2_round(dims):
    return 2 ** torch.round(torch.log2(dims.type(torch.float)))


class PrEmbeddingBag(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, base_dim):
        super(PrEmbeddingBag, self).__init__()
        self.embs = nn.EmbeddingBag(
            num_embeddings, embedding_dim, mode="sum", sparse=False)
        torch.nn.init.xavier_uniform_(self.embs.weight)
        if embedding_dim < base_dim:
            self.proj = nn.Linear(embedding_dim, base_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.proj.weight)
        elif embedding_dim == base_dim:
            self.proj = nn.Identity()
        else:
            raise ValueError(
                "Embedding dim " + str(embedding_dim) + " > base dim " + str(base_dim)
            )

    def forward(self, input, offsets=None, per_sample_weights=None):
        offsets = input.new_zeros(1)
        return self.proj(self.embs(input, offsets=offsets, per_sample_weights=per_sample_weights))

def test():
    m_spa = args.arch_sparse_feature_size
    m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims,
            k=16,
        ).tolist()
    # pdb.set_trace()
    emb_l = create_emb(m_spa, ln_emb)
    pdb.set_trace()

def create_emb(m, ln):
    emb_l = nn.ModuleList()
    for i in range(0, ln.size):
        n = ln[i]
        _m = m[i]
        base = max(m)
        EE = PrEmbeddingBag(n, _m, base)
        # use np initialization as below for consistency...
        W = np.random.uniform(
            low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
        ).astype(np.float32)
        EE.embs.weight.data = torch.tensor(W, requires_grad=True)
        emb_l.append(EE)
    return emb_l


def apply_emb(lS_o, lS_i, emb_l):
    ly = []
    for k, sparse_index_group_batch in enumerate(lS_i):
        sparse_offset_group_batch = lS_o[k]
        E = emb_l[k]
        V = E(sparse_index_group_batch, sparse_offset_group_batch)
        ly.append(V)
    return ly