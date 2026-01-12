<div align="center">
  <h1>EasyFlow - WIP</h1>
  <p><strong>Quality-of-life utilities for <a href="https://github.com/The-Pocket/PocketFlow">PocketFlow</a></strong></p>
</div>

<br>

EasyFlow extends [PocketFlow](https://github.com/The-Pocket/PocketFlow)—the 100-line minimalist LLM framework—with optional utilities that address common pain points while preserving the simplistic philosophy.

## What's Included

### LLM Provider Router

One function per provider. All models. No conditionals in your nodes.

```python
from easyflow.utils import register_llm, call_llm

# Register once per provider
def openai_call(prompt, model, **kwargs):
    client = OpenAI()
    r = client.chat.completions.create(
        model=model or "gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )
    return r.choices[0].message.content

register_llm("openai", openai_call)
register_llm("anthropic", anthropic_call)

# Use any model with a clean API
call_llm("Hello!", "openai/gpt-4o")
call_llm("Hello!", "openai/gpt-4o-mini")
call_llm("Hello!", "anthropic/claude-sonnet-4-0")
```

### Embedding Provider Router

Same pattern for embeddings:

```python
from easyflow.utils import register_embedding, embed

register_embedding("openai", openai_embed)

embed("Hello world", "openai/text-embedding-3-small")
embed("Hello world", "openai/text-embedding-3-large")
```

## Learn PocketFlow

EasyFlow is built on PocketFlow. For core concepts—Nodes, Flows, Agents, RAG, etc.—see:

- 📖 [PocketFlow Documentation](https://the-pocket.github.io/PocketFlow/)
- 🎥 [Video Tutorial](https://youtu.be/0Zr3NwcvpA0)
- 💬 [Discord Community](https://discord.gg/hUHHE9Sa6T)
- 📦 [PocketFlow GitHub](https://github.com/The-Pocket/PocketFlow)

## Philosophy

PocketFlow is intentionally minimal—100 lines, zero dependencies, zero vendor lock-in.

EasyFlow shares that philosophy. We add utilities only when they:

- Eliminate repetitive boilerplate
- Stay dependency-free
- Remain optional (use what you need, ignore the rest)

The utilities here are things we found ourselves writing over and over. Now you don't have to.

## License

MIT
