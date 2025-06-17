from setuptools import setup

setup(name='llmbridge',
      version='0.0.1',
      author='Top Piriyakulkij, zzigak',
      packages=['llmbridge'],
      description="A unified interface that bridges different LLM providers (OpenAI, HuggingFace, etc.) with a common API.",
      license='MIT',
      install_requires=[
        'openai>=1.0',
        'sqlalchemy',
        'tiktoken',
        'numpy'
      ],
)