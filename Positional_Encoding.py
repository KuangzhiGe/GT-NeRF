import torch

class Embedder:
    def __init__(self, input_dims=3, include_input=True, L=10):
        self.input_dims = input_dims
        self.include_input = include_input
        self.L = L
        self.create_embedding_fn()
    
    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        
        if self.include_input:
            embed_fns.append(lambda x : x)
            out_dim += d
        
        freq_bands = 2. ** torch.linspace(0., self.L - 1, steps=self.L)
        
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, : p_fn(x * freq))
                out_dim += d
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(L=10, input_dims=3, include_input=True):
    embedder_obj = Embedder(input_dims=input_dims, include_input=include_input, L=L)
    embed = lambda x, embedder_obj = embedder_obj : embedder_obj.embed(x)
    return embed, embedder_obj.out_dim