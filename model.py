import jax.numpy as jnp
import jax
import numpy as np
import flax.nnx as nnx
import optax
import logging, os, time


class GPTConfig:
    batch_size: int = 64
    hidden_size: int = 384
    n_layers: int = 6
    n_heads: int = 6
    vocab_size: int = 65 
    learning_rate: float = 1e-4
    context_length: int = 256
    max_iters: int = 5000
    eval_interval: int = 10
class LinearLayer(nnx.Linear):
    def __init__(self, in_features, out_features, use_bias, rngs: nnx.Rngs):
        super().__init__(in_features, out_features, use_bias=use_bias, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, C = x.shape
        x = x.reshape(-1, x.shape[-1])
        return super().__call__(x).reshape(B, T, -1)
    
class SelfAttention(nnx.Module):
    def __init__(self, hidden_size, n_heads, rngs: nnx.Rngs):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        assert self.hidden_size % self.n_heads == 0, "hidden_size must be divisible by n_heads"

        self.qkv = LinearLayer(self.hidden_size, 3 * self.hidden_size, use_bias=False, rngs=rngs)
        self.proj = LinearLayer(self.hidden_size, self.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3 *):

        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_heads, self.head_dim).swapaxes(1, 2) # B, n_heads, T, head_dim
        k = k.reshape(B, T, self.n_heads, self.head_dim).swapaxes(1, 2) # B, n_heads, T, head_dim
        v = v.reshape(B, T, self.n_heads, self.head_dim).swapaxes(1, 2) # B, n_heads, T, head_dim

        mat = q @ k.swapaxes(-2, -1) * 1.0 / jnp.sqrt(self.head_dim)
        mask = jnp.tril(jnp.ones((1,1, T, T), dtype=jnp.bool))
        mat = jnp.where(mask, mat, -jnp.inf)
        mat = jax.nn.softmax(mat, axis=-1)
        x = mat @ v # B, T, n_heads, head_dim
        x = x.swapaxes(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x
class SwiGLU(nnx.Module):
    def __init__(self, dim, hidden_dim, rngs: nnx.Rngs):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.gate = LinearLayer(dim, hidden_dim, use_bias=False, rngs=rngs)
        self.up = LinearLayer(dim, hidden_dim, use_bias=False, rngs=rngs)
        self.down = LinearLayer(hidden_dim, dim, use_bias=False, rngs=rngs)
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))
    
class MLP(nnx.Module):
    def __init__(self, hidden_size, rngs: nnx.Rngs):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = LinearLayer(self.hidden_size, 4 * self.hidden_size, use_bias=False, rngs=rngs)
        self.fc2 = LinearLayer(4 * self.hidden_size, self.hidden_size, use_bias=False, rngs=rngs)
        #self.swiglu = SwiGLU(self.hidden_size, self.hidden_size, rngs=rngs)
        #self.gelu = nnx.gelu()

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x

class Block(nnx.Module):
    def __init__(self, hidden_size, n_heads, rngs: nnx.Rngs):
        self.attn = SelfAttention(hidden_size, n_heads, rngs=rngs)
        self.mlp = MLP(hidden_size, rngs=rngs)
        self.norm1 = nnx.RMSNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.norm2 = nnx.RMSNorm(hidden_size, epsilon=1e-6, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x   

class Model(nnx.Module):
    def __init__(self, hidden_size, n_heads, n_layers, vocab_size, context_length, rngs: nnx.Rngs):
        self.blocks = nnx.List([Block(hidden_size, n_heads, rngs=rngs) 
                                for _ in range(n_layers)])
        self.norm = nnx.RMSNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.embedding = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
        self.pembedding = nnx.Embed(context_length, hidden_size, rngs=rngs)
        self.head = LinearLayer(hidden_size, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, idx: jax.Array) -> jax.Array:
        B, T = idx.shape
        x = self.embedding(idx) + self.pembedding(jnp.arange(T))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        return x
@nnx.jit
def cross_entropy_loss(model, x, labels):
    logits = model(x)
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss

@nnx.jit
def train_step(model, optimizer, x, labels):
    loss, grads = nnx.value_and_grad(cross_entropy_loss)(model, x, labels)
    optimizer.update(model, grads)
    return loss


def get_data(split, rngs):
    #key, subkey = jax.random.split(key)
    if split == 'train':
        data = np.memmap(os.path.join('./data/shakespeare_char', "train.bin"),dtype=np.uint16, mode='r')
    else: 
        data = np.memmap(os.path.join('./data/shakespeare_char', "val.bin"),dtype=np.uint16)
    idices = jax.random.randint(rngs(), shape=(config.batch_size,), 
                                minval=0, maxval=len(data)-config.context_length)
    x = jnp.stack([jnp.asarray(data[idx:idx+config.context_length]) 
                for idx in idices])
    y = jnp.stack([jnp.asarray(data[idx+1:idx+1+config.context_length]) 
                for idx in idices])
    #import pdb; pdb.set_trace()
    return (x, y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_rngs = nnx.Rngs(42)
    data_rngs = nnx.Rngs(43)
    config = GPTConfig()
    model = Model(config.hidden_size, config.n_heads, config.n_layers,
                  config.vocab_size, config.context_length, rngs=model_rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=config.learning_rate), 
                              wrt=nnx.Param)
    params = nnx.state(model, nnx.Param)
    num_params = sum(p.size for p in jax.tree.leaves(params))
    logging.info(f"number of parameters: {num_params/1e6:.2f}M")

    #import pdb; pdb.set_trace()
    for path, p in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = jax.tree_util.keystr(path)
        
        logging.info(f"{name}: shape={p.shape}, dtype={p.dtype},count={p.size}")

    iter = 0
    t0=time.time()
    while iter < config.max_iters:
        iter += 1
        x, y = get_data('train', data_rngs)
        loss = train_step(model, optimizer, x, y)
        t1 = time.time()

        if iter % config.eval_interval == 0:
            x, y = get_data('val', data_rngs)
            val = cross_entropy_loss(model, x, y)
            logging.info(f"iter {iter}: loss {loss:.4f}, val loss {val:.4f}, time {(t1-t0)*1000:.4f}ms")
        else:
            logging.info(f"iter {iter}: loss {loss:.4f}, time {(t1-t0)*1000:.4f}ms")
        t0 = t1


    
