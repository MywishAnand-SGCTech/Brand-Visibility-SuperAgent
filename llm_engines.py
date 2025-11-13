import os
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import httpx

@dataclass
class LLMResponse:
    engine: str
    response: str
    metadata: Dict[str, Any]
    error: Optional[str] = None

class LLMEngineManager:
    def __init__(self):
        self.engines = {}
        self.setup_engines()
    
    def setup_engines(self):
        """Initialize all LLM engines with their configurations"""
        # Free LLMs
        self.engines['ollama'] = {
            'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            'api_key': os.getenv('OLLAMA_API_KEY', ''),
            'model': os.getenv('OLLAMA_MODEL', 'llama2'),
            'enabled': True
        }
        
        self.engines['deepseek'] = {
            'base_url': os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            'api_key': os.getenv('DEEPSEEK_API_KEY', ''),
            'model': os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
            'enabled': True
        }
        
        self.engines['groq'] = {
            'base_url': os.getenv('GROQ_BASE_URL', 'https://api.groq.com'),
            'api_key': os.getenv('GROQ_API_KEY', ''),
            'model': os.getenv('GROQ_MODEL', 'llama3-70b-8192'),
            'enabled': True
        }
        
        self.engines['mistralai'] = {
            'base_url': os.getenv('MISTRAL_BASE_URL', 'https://api.mistral.ai'),
            'api_key': os.getenv('MISTRAL_API_KEY', ''),
            'model': os.getenv('MISTRAL_MODEL', 'mistral-medium'),
            'enabled': True
        }
        
        self.engines['huggingface'] = {
            'base_url': os.getenv('HF_BASE_URL', 'https://api-inference.huggingface.co'),
            'api_key': os.getenv('HF_API_KEY', ''),
            'model': os.getenv('HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.1'),
            'enabled': True
        }
        
        # Paid LLMs
        self.engines['gemini'] = {
            'base_url': os.getenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com'),
            'api_key': os.getenv('GEMINI_API_KEY', ''),
            'model': os.getenv('GEMINI_MODEL', 'gemini-pro'),
            'enabled': True
        }
        
        self.engines['openai'] = {
            'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com'),
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'model': os.getenv('OPENAI_MODEL', 'gpt-4'),
            'enabled': True
        }
        
        self.engines['anthropic'] = {
            'base_url': os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
            'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
            'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
            'enabled': True
        }
        
        self.engines['perplexity'] = {
            'base_url': os.getenv('PERPLEXITY_BASE_URL', 'https://api.perplexity.ai'),
            'api_key': os.getenv('PERPLEXITY_API_KEY', ''),
            'model': os.getenv('PERPLEXITY_MODEL', 'sonar-medium-online'),
            'enabled': True
        }

    async def call_ollama(self, prompt: str, engine_config: Dict) -> LLMResponse:
        """Call Ollama API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": engine_config['model'],
                    "prompt": prompt,
                    "stream": False
                }
                async with session.post(
                    f"{engine_config['base_url']}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return LLMResponse(
                            engine="ollama",
                            response=data.get('response', ''),
                            metadata={'model': engine_config['model']}
                        )
                    else:
                        return LLMResponse(
                            engine="ollama",
                            response="",
                            metadata={},
                            error=f"HTTP {response.status}"
                        )
        except Exception as e:
            return LLMResponse(
                engine="ollama",
                response="",
                metadata={},
                error=str(e)
            )

    async def call_openai_compatible(self, prompt: str, engine_config: Dict, engine_name: str) -> LLMResponse:
        """Call OpenAI-compatible APIs"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {engine_config['api_key']}"
            }
            
            payload = {
                "model": engine_config['model'],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{engine_config['base_url']}/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return LLMResponse(
                            engine=engine_name,
                            response=content,
                            metadata={
                                'model': engine_config['model'],
                                'usage': data.get('usage', {})
                            }
                        )
                    else:
                        return LLMResponse(
                            engine=engine_name,
                            response="",
                            metadata={},
                            error=f"HTTP {response.status}"
                        )
        except Exception as e:
            return LLMResponse(
                engine=engine_name,
                response="",
                metadata={},
                error=str(e)
            )

    async def call_gemini(self, prompt: str, engine_config: Dict) -> LLMResponse:
        """Call Gemini API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{engine_config['base_url']}/v1/models/{engine_config['model']}:generateContent?key={engine_config['api_key']}"
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                }
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        return LLMResponse(
                            engine="gemini",
                            response=content,
                            metadata={'model': engine_config['model']}
                        )
                    else:
                        return LLMResponse(
                            engine="gemini",
                            response="",
                            metadata={},
                            error=f"HTTP {response.status}"
                        )
        except Exception as e:
            return LLMResponse(
                engine="gemini",
                response="",
                metadata={},
                error=str(e)
            )

    async def call_anthropic(self, prompt: str, engine_config: Dict) -> LLMResponse:
        """Call Anthropic Claude API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": engine_config['api_key'],
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": engine_config['model'],
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{engine_config['base_url']}/v1/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['content'][0]['text']
                        return LLMResponse(
                            engine="anthropic",
                            response=content,
                            metadata={
                                'model': engine_config['model'],
                                'usage': data.get('usage', {})
                            }
                        )
                    else:
                        return LLMResponse(
                            engine="anthropic",
                            response="",
                            metadata={},
                            error=f"HTTP {response.status}"
                        )
        except Exception as e:
            return LLMResponse(
                engine="anthropic",
                response="",
                metadata={},
                error=str(e)
            )

    async def call_engine(self, prompt: str, engine_name: str) -> LLMResponse:
        """Route to appropriate engine handler"""
        engine_config = self.engines.get(engine_name)
        if not engine_config or not engine_config['enabled']:
            return LLMResponse(
                engine=engine_name,
                response="",
                metadata={},
                error="Engine disabled or not configured"
            )
        
        if engine_name == 'ollama':
            return await self.call_ollama(prompt, engine_config)
        elif engine_name == 'gemini':
            return await self.call_gemini(prompt, engine_config)
        elif engine_name == 'anthropic':
            return await self.call_anthropic(prompt, engine_config)
        else:
            # OpenAI-compatible APIs
            return await self.call_openai_compatible(prompt, engine_config, engine_name)

    async def call_all_engines_parallel(self, prompt: str) -> Dict[str, LLMResponse]:
        """Call all enabled engines in parallel"""
        tasks = []
        for engine_name in self.engines:
            if self.engines[engine_name]['enabled']:
                task = self.call_engine(prompt, engine_name)
                tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return {resp.engine: resp for resp in responses if resp.engine}

# Global instance
llm_manager = LLMEngineManager()
