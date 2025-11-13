import json
import re
import asyncio
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from urllib.parse import urlparse
import openai
from langchain.schema import BaseOutputParser
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
import numpy as np

@dataclass
class CompanyMention:
    name: str
    canonical_name: str
    mentions: int
    sources: List[Dict[str, Any]]
    confidence: float
    citations: List[str]

class Agent1_Researcher:
    """Agent 1: Expert researcher that crafts prompt questions"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.base_prompt_template = """
        You are an expert market researcher and business analyst. For the given search query "{query}" in location "{location}", 
        create comprehensive prompt questions that will help identify all companies, corporations, and businesses that are 
        contextually and competitively associated with this query.
        
        Requirements:
        1. Generate 5-10 specific, targeted prompt questions
        2. Questions should be designed to extract company names from any LLM
        3. Focus on competitive landscape, market players, and industry participants
        4. Include questions about direct competitors, related services, and industry leaders
        5. Make questions actionable for LLM engines
        
        Format: Return as a JSON list of questions.
        """
    
    async def generate_research_questions(self, query: str, location: str) -> Dict[str, Any]:
        """Generate research questions using all LLM engines in parallel"""
        prompt = self.base_prompt_template.format(query=query, location=location)
        
        # Call all engines in parallel
        responses = await self.llm_manager.call_all_engines_parallel(prompt)
        
        # Parse questions from each engine
        engine_questions = {}
        all_questions = set()
        
        for engine_name, response in responses.items():
            if response.response:
                try:
                    questions = self._parse_questions_from_response(response.response)
                    engine_questions[engine_name] = questions
                    all_questions.update(questions)
                except Exception as e:
                    print(f"Error parsing questions from {engine_name}: {e}")
                    engine_questions[engine_name] = []
        
        return {
            'engine_questions': engine_questions,
            'all_unique_questions': list(all_questions),
            'raw_responses': {k: v.response for k, v in responses.items()}
        }
    
    def _parse_questions_from_response(self, response: str) -> List[str]:
        """Parse questions from LLM response"""
        # Try JSON parsing first
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return [str(q) for q in data if q]
        except:
            pass
        
        # Fallback: extract questions using regex
        questions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('?' in line or line.startswith('1.') or line.startswith('-')): 
                # Clean the line
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                clean_line = re.sub(r'^[-*]\s*', '', clean_line)
                clean_line = clean_line.strip()
                if clean_line and len(clean_line) > 10:  # Minimum length for meaningful question
                    questions.append(clean_line)
        
        return questions[:10]  # Limit to 10 questions

class Agent2_QuestionCollector:
    """Agent 2: Collects and consolidates questions from all engines"""
    
    def __init__(self):
        self.collected_questions = {}
    
    def process_questions(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and consolidate questions from all engines"""
        engine_questions = research_results['engine_questions']
        all_unique_questions = research_results['all_unique_questions']
        
        # Save engine-wise questions
        self._save_engine_wise_questions(engine_questions)
        
        return {
            'consolidated_questions': all_unique_questions,
            'engine_breakdown': engine_questions,
            'total_unique_questions': len(all_unique_questions)
        }
    
    def _save_engine_wise_questions(self, engine_questions: Dict[str, List[str]]):
        """Save engine-wise questions to JSON"""
        output_data = {}
        for engine, questions in engine_questions.items():
            output_data[engine] = {
                'question_count': len(questions),
                'questions': questions
            }
        
        with open('engine_wise_questions.json', 'w') as f:
            json.dump(output_data, f, indent=2)

class Agent3_QuestionExecutor:
    """Agent 3: Executes all questions on all LLM engines"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.results = {}
    
    async def execute_all_questions(self, questions: List[str]) -> Dict[str, Any]:
        """Execute all questions on all LLM engines in parallel"""
        all_results = {}
        
        for question in questions:
            question_results = await self.llm_manager.call_all_engines_parallel(question)
            all_results[question] = question_results
            
            # Save intermediate results
            self._save_question_responses(question, question_results)
        
        self.results = all_results
        return all_results
    
    def _save_question_responses(self, question: str, responses: Dict[str, Any]):
        """Save responses for each question"""
        question_data = {
            'question': question,
            'responses': {}
        }
        
        for engine_name, response in responses.items():
            question_data['responses'][engine_name] = {
                'response': response.response,
                'metadata': response.metadata,
                'error': response.error
            }
        
        # Append to file or create new
        try:
            with open('question_responses.json', 'r') as f:
                existing_data = json.load(f)
        except:
            existing_data = {}
        
        existing_data[question] = question_data
        
        with open('question_responses.json', 'w') as f:
            json.dump(existing_data, f, indent=2)

class Agent4_CompanyExtractor:
    """Agent 4: Extract company names from responses using guardrails and NLP"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            print("spaCy model not available, using fallback methods")
        
        # Company suffixes for guardrails
        self.company_suffixes = {
            'ltd', 'limited', 'llc', 'inc', 'corporation', 'corp', 'co', 'company',
            'gmbh', 'ag', 'sa', 'nv', 'plc', 'pty', 'ab', 'oy', 'as', 'spa'
        }
        
        # Common exclusion terms
        self.exclusion_terms = {
            'test', 'example', 'sample', 'demo', 'placeholder', 'fictional',
            'fake', 'dummy', 'mock', 'prototype'
        }
    
    def extract_companies_from_responses(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract companies from all responses with guardrails"""
        all_companies = {}
        
        for question, engine_responses in execution_results.items():
            for engine_name, response_obj in engine_responses.items():
                if response_obj.response:
                    companies = self._extract_companies_from_text(
                        response_obj.response, 
                        question, 
                        engine_name
                    )
                    
                    for company in companies:
                        if company['name'] not in all_companies:
                            all_companies[company['name']] = {
                                'canonical_name': company['canonical_name'],
                                'mentions': 0,
                                'sources': [],
                                'confidence_scores': [],
                                'citations': company['citations']
                            }
                        
                        all_companies[company['name']]['mentions'] += 1
                        all_companies[company['name']]['sources'].append({
                            'engine': engine_name,
                            'question': question,
                            'context': company['context_snippet']
                        })
                        all_companies[company['name']]['confidence_scores'].append(
                            company['confidence']
                        )
        
        # Calculate average confidence scores
        for company in all_companies.values():
            if company['confidence_scores']:
                company['avg_confidence'] = sum(company['confidence_scores']) / len(company['confidence_scores'])
            else:
                company['avg_confidence'] = 0
        
        return all_companies
    
    def _extract_companies_from_text(self, text: str, question: str, engine: str) -> List[Dict[str, Any]]:
        """Extract companies from a single text response"""
        companies = []
        
        # Method 1: spaCy NER
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT']:
                    company_data = self._validate_company(ent.text, text)
                    if company_data:
                        companies.append(company_data)
        
        # Method 2: Regex patterns for company names
        company_patterns = [
            r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]*)*(?:\s+(?:{})\.?\b)'.format('|'.join(self.company_suffixes)),
            r'\b(?:[A-Z][a-z]+\s+){1,3}(?:Inc|LLC|Ltd|Corp|Co)\.?\b',
        ]
        
        for pattern in company_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                company_name = match.group().strip()
                company_data = self._validate_company(company_name, text)
                if company_data:
                    companies.append(company_data)
        
        # Method 3: TF-IDF based extraction (simplified)
        companies.extend(self._tfidf_company_extraction(text))
        
        return companies
    
    def _validate_company(self, candidate: str, context: str) -> Dict[str, Any]:
        """Validate if a candidate string is actually a company"""
        candidate_lower = candidate.lower()
        
        # Exclusion checks
        if any(excl in candidate_lower for excl in self.exclusion_terms):
            return None
        
        # Check for company indicators
        has_suffix = any(suffix in candidate_lower for suffix in self.company_suffixes)
        has_capital_words = len([w for w in candidate.split() if w and w[0].isupper()]) >= 2
        
        confidence = 0.0
        if has_suffix:
            confidence += 0.6
        if has_capital_words:
            confidence += 0.4
        
        if confidence > 0.5:
            # Extract citations/links
            citations = self._extract_citations(context)
            
            return {
                'name': candidate,
                'canonical_name': self._canonicalize_company_name(candidate),
                'confidence': min(confidence, 1.0),
                'context_snippet': context[:200] + '...' if len(context) > 200 else context,
                'citations': citations
            }
        
        return None
    
    def _canonicalize_company_name(self, name: str) -> str:
        """Canonicalize company name by removing suffixes and standardizing"""
        # Remove common suffixes
        canonical = re.sub(r'\s+(?:{})\.?\s*$'.format('|'.join(self.company_suffixes)), '', name, flags=re.IGNORECASE)
        return canonical.strip()
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract URLs and citations from text"""
        url_pattern = r'https?://[^\s]+|www\.[^\s]+'
        urls = re.findall(url_pattern, text)
        return urls
    
    def _tfidf_company_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Simple TF-IDF based company extraction"""
        # This is a simplified version - in production you'd use proper TF-IDF
        # across all documents to identify important entity terms
        words = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)
        companies = []
        
        for word_pair in words[:5]:  # Limit to top 5
            company_data = self._validate_company(word_pair, text)
            if company_data:
                companies.append(company_data)
        
        return companies

class Agent5_ScoringAnalyzer:
    """Agent 5: Score and analyze company mentions"""
    
    def __init__(self):
        self.domain_authority_scores = {}
    
    def analyze_companies(self, extracted_companies: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and score companies based on mentions and citations"""
        scored_companies = {}
        
        for company_name, company_data in extracted_companies.items():
            # Base visibility score (mention frequency)
            visibility_score = min(company_data['mentions'] / 10, 1.0)  # Normalize
            
            # Confidence score (average of confidence scores)
            confidence_score = company_data.get('avg_confidence', 0)
            
            # Citation quality score
            citation_score = self._calculate_citation_score(company_data['citations'])
            
            # Engine consensus score
            engine_consensus = self._calculate_engine_consensus(company_data['sources'])
            
            # Final composite score
            final_score = (
                visibility_score * 0.3 +
                confidence_score * 0.3 +
                citation_score * 0.2 +
                engine_consensus * 0.2
            )
            
            scored_companies[company_name] = {
                **company_data,
                'visibility_score': visibility_score,
                'citation_score': citation_score,
                'engine_consensus': engine_consensus,
                'final_score': final_score,
                'source_count': len(company_data['sources'])
            }
        
        # Rank companies by final score
        ranked_companies = dict(sorted(
            scored_companies.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        ))
        
        return ranked_companies
    
    def _calculate_citation_score(self, citations: List[str]) -> float:
        """Calculate citation quality score based on domain authority"""
        if not citations:
            return 0.0
        
        total_score = 0.0
        for citation in citations:
            domain = self._extract_domain(citation)
            authority = self._get_domain_authority(domain)
            total_score += authority
        
        return total_score / len(citations)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url if url.startswith('http') else f'http://{url}')
            return parsed.netloc
        except:
            return ""
    
    def _get_domain_authority(self, domain: str) -> float:
        """Simple domain authority scoring (simplified)"""
        # In production, integrate with Moz DA or similar service
        high_authority_domains = {
            'forbes.com', 'bloomberg.com', 'reuters.com', 'wsj.com',
            'techcrunch.com', 'wired.com', 'nytimes.com'
        }
        
        medium_authority_domains = {
            'medium.com', 'linkedin.com', 'crunchbase.com', 'angel.co'
        }
        
        if domain in high_authority_domains:
            return 1.0
        elif domain in medium_authority_domains:
            return 0.7
        elif domain.endswith(('.com', '.org', '.net')):
            return 0.5
        else:
            return 0.3
    
    def _calculate_engine_consensus(self, sources: List[Dict]) -> float:
        """Calculate how many different engines mentioned the company"""
        unique_engines = set(source['engine'] for source in sources)
        return len(unique_engines) / 10  # Normalize based on expected max engines

class Agent6_ReportGenerator:
    """Agent 6: Generate final reports and statistics"""
    
    def generate_reports(self, scored_companies: Dict[str, Any], 
                        research_data: Dict[str, Any],
                        execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all final reports"""
        
        # 1. Prompt-specific JSON
        prompt_specific_data = self._generate_prompt_specific_json(execution_data)
        
        # 2. Company-specific JSON
        company_specific_data = self._generate_company_specific_json(scored_companies)
        
        # 3. Engine-specific JSON
        engine_specific_data = self._generate_engine_specific_json(execution_data, scored_companies)
        
        # 4. CSV Report
        self._generate_csv_report(scored_companies)
        
        return {
            'prompt_specific': prompt_specific_data,
            'company_specific': company_specific_data,
            'engine_specific': engine_specific_data
        }
    
    def _generate_prompt_specific_json(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prompt-specific analysis"""
        prompt_stats = {}
        
        for question, engines in execution_data.items():
            prompt_stats[question] = {
                'total_engines_responded': len(engines),
                'total_companies_found': 0,  # Would need company mapping
                'average_response_length': np.mean([
                    len(resp.response) for resp in engines.values() if resp.response
                ])
            }
        
        with open('prompt_specific_analysis.json', 'w') as f:
            json.dump(prompt_stats, f, indent=2)
        
        return prompt_stats
    
    def _generate_company_specific_json(self, scored_companies: Dict[str, Any]) -> Dict[str, Any]:
        """Generate company-specific analysis with all references"""
        company_data = {}
        
        for company_name, company_info in scored_companies.items():
            company_data[company_name] = {
                'canonical_name': company_info['canonical_name'],
                'final_score': company_info['final_score'],
                'visibility_score': company_info['visibility_score'],
                'citation_score': company_info['citation_score'],
                'engine_consensus': company_info['engine_consensus'],
                'total_mentions': company_info['mentions'],
                'sources': company_info['sources'],
                'citations': company_info['citations'],
                'confidence_history': company_info['confidence_scores']
            }
        
        with open('company_specific_analysis.json', 'w') as f:
            json.dump(company_data, f, indent=2)
        
        return company_data
    
    def _generate_engine_specific_json(self, execution_data: Dict[str, Any], 
                                     scored_companies: Dict[str, Any]) -> Dict[str, Any]:
        """Generate engine-specific analysis"""
        engine_stats = defaultdict(lambda: {
            'total_questions_answered': 0,
            'total_companies_found': 0,
            'average_confidence': 0.0,
            'total_responses': 0
        })
        
        # Count engine performance
        for question, engines in execution_data.items():
            for engine_name, response in engines.items():
                if response.response:
                    engine_stats[engine_name]['total_questions_answered'] += 1
                    engine_stats[engine_name]['total_responses'] += 1
        
        with open('engine_specific_analysis.json', 'w') as f:
            json.dump(dict(engine_stats), f, indent=2)
        
        return dict(engine_stats)
    
    def _generate_csv_report(self, scored_companies: Dict[str, Any]):
        """Generate CSV report for companies"""
        import csv
        
        with open('brand_visibility_report.csv', 'w', newline='') as csvfile:
            fieldnames = [
                'Company Name', 'Canonical Name', 'Final Score', 'Visibility Score',
                'Citation Score', 'Engine Consensus', 'Total Mentions', 'Source Count'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for company_name, company_info in scored_companies.items():
                writer.writerow({
                    'Company Name': company_name,
                    'Canonical Name': company_info['canonical_name'],
                    'Final Score': round(company_info['final_score'], 4),
                    'Visibility Score': round(company_info['visibility_score'], 4),
                    'Citation Score': round(company_info['citation_score'], 4),
                    'Engine Consensus': round(company_info['engine_consensus'], 4),
                    'Total Mentions': company_info['mentions'],
                    'Source Count': company_info['source_count']
                })

class BrandVisibilitySuperAgent:
    """Main SuperAgent coordinating all agents"""
    
    def __init__(self, llm_manager):
        self.agent1 = Agent1_Researcher(llm_manager)
        self.agent2 = Agent2_QuestionCollector()
        self.agent3 = Agent3_QuestionExecutor(llm_manager)
        self.agent4 = Agent4_CompanyExtractor()
        self.agent5 = Agent5_ScoringAnalyzer()
        self.agent6 = Agent6_ReportGenerator()
    
    async def run_analysis(self, query: str, location: str) -> Dict[str, Any]:
        """Run complete brand visibility analysis"""
        print("Starting Brand Visibility Analysis...")
        
        # Agent 1: Generate research questions
        print("Agent 1: Generating research questions...")
        research_results = await self.agent1.generate_research_questions(query, location)
        
        # Agent 2: Collect and consolidate questions
        print("Agent 2: Consolidating questions...")
        question_data = self.agent2.process_questions(research_results)
        
        # Agent 3: Execute all questions
        print("Agent 3: Executing questions on all engines...")
        execution_results = await self.agent3.execute_all_questions(
            question_data['consolidated_questions']
        )
        
        # Agent 4: Extract companies
        print("Agent 4: Extracting companies from responses...")
        extracted_companies = self.agent4.extract_companies_from_responses(execution_results)
        
        # Agent 5: Score companies
        print("Agent 5: Scoring and analyzing companies...")
        scored_companies = self.agent5.analyze_companies(extracted_companies)
        
        # Agent 6: Generate reports
        print("Agent 6: Generating final reports...")
        final_reports = self.agent6.generate_reports(
            scored_companies, research_results, execution_results
        )
        
        print("Analysis complete! Check generated JSON and CSV files.")
        
        return {
            'research_questions': question_data,
            'execution_results': execution_results,
            'scored_companies': scored_companies,
            'final_reports': final_reports
        }
