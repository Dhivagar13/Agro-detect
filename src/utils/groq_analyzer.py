"""Groq AI-powered analysis for enhanced disease insights"""

import os
import json
from typing import Dict, Optional
import requests


class GroqAnalyzer:
    """
    Enhanced AI analysis using Groq LLM for contextual disease insights
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Groq Analyzer
        
        Args:
            api_key: Groq API key
        """
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "mixtral-8x7b-32768"  # Fast and accurate model
    
    def analyze_disease(
        self,
        disease_name: str,
        confidence: float,
        symptoms: list,
        causes: list,
        context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get AI-powered analysis and recommendations
        
        Args:
            disease_name: Detected disease name
            confidence: Detection confidence
            symptoms: List of symptoms
            causes: List of causes
            context: Additional context (optional)
        
        Returns:
            Dictionary with AI analysis
        """
        
        prompt = self._build_prompt(disease_name, confidence, symptoms, causes, context)
        
        try:
            response = self._call_groq_api(prompt)
            return self._parse_response(response)
        except Exception as e:
            return {
                "analysis": f"AI analysis unavailable: {str(e)}",
                "recommendations": "Please refer to the standard treatment guidelines.",
                "urgency": "moderate",
                "additional_tips": ""
            }
    
    def _build_prompt(
        self,
        disease_name: str,
        confidence: float,
        symptoms: list,
        causes: list,
        context: Optional[str]
    ) -> str:
        """Build the prompt for Groq API"""
        
        prompt = f"""You are an expert agricultural pathologist and plant disease specialist. Analyze the following plant disease detection and provide actionable insights.

**Disease Detected:** {disease_name}
**Detection Confidence:** {confidence:.1f}%
**Observed Symptoms:** {', '.join(symptoms) if symptoms else 'Not specified'}
**Known Causes:** {', '.join(causes) if causes else 'Not specified'}
{f'**Additional Context:** {context}' if context else ''}

Please provide a comprehensive analysis in the following format:

1. **ANALYSIS** (2-3 sentences):
   - Assess the severity and urgency
   - Explain the disease progression
   - Consider the confidence level

2. **IMMEDIATE RECOMMENDATIONS** (3-5 actionable steps):
   - Prioritized actions the farmer should take NOW
   - Be specific and practical
   - Consider both organic and chemical options

3. **URGENCY LEVEL** (one word only):
   - low / moderate / high / critical

4. **ADDITIONAL TIPS** (2-3 practical tips):
   - Weather considerations
   - Monitoring advice
   - Prevention for future crops

Keep your response concise, practical, and farmer-friendly. Focus on actionable advice."""

        return prompt
    
    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert agricultural pathologist specializing in plant disease diagnosis and treatment. Provide clear, actionable advice to farmers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1,
            "stream": False
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse the AI response into structured format"""
        
        # Initialize default values
        analysis = ""
        recommendations = ""
        urgency = "moderate"
        additional_tips = ""
        
        # Parse the response
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            # Detect sections
            if 'ANALYSIS' in line.upper():
                current_section = 'analysis'
                continue
            elif 'IMMEDIATE RECOMMENDATION' in line.upper() or 'RECOMMENDATION' in line.upper():
                current_section = 'recommendations'
                continue
            elif 'URGENCY' in line.upper():
                current_section = 'urgency'
                continue
            elif 'ADDITIONAL TIP' in line.upper() or 'TIP' in line.upper():
                current_section = 'additional_tips'
                continue
            
            # Add content to appropriate section
            if current_section == 'analysis':
                if line and not line.startswith('**'):
                    analysis += line + " "
            elif current_section == 'recommendations':
                if line and not line.startswith('**'):
                    recommendations += line + "\n"
            elif current_section == 'urgency':
                # Extract urgency level
                for word in ['critical', 'high', 'moderate', 'low']:
                    if word in line.lower():
                        urgency = word
                        break
            elif current_section == 'additional_tips':
                if line and not line.startswith('**'):
                    additional_tips += line + "\n"
        
        return {
            "analysis": analysis.strip() or "AI analysis completed successfully.",
            "recommendations": recommendations.strip() or "Follow standard treatment protocols.",
            "urgency": urgency,
            "additional_tips": additional_tips.strip() or "Monitor plant health regularly."
        }
    
    def get_weather_advice(self, disease_name: str, location: Optional[str] = None) -> str:
        """
        Get weather-specific advice for disease management
        
        Args:
            disease_name: Disease name
            location: Location (optional)
        
        Returns:
            Weather-specific advice
        """
        
        prompt = f"""As an agricultural expert, provide weather-specific advice for managing {disease_name}.
        
{f'Location: {location}' if location else ''}

Provide 2-3 specific weather-related tips for:
1. When to apply treatments
2. Weather conditions to avoid
3. Optimal conditions for recovery

Keep it brief and actionable."""

        try:
            response = self._call_groq_api(prompt)
            return response.strip()
        except Exception as e:
            return "Monitor weather conditions and apply treatments during dry periods."
    
    def compare_treatments(
        self,
        disease_name: str,
        organic_options: list,
        chemical_options: list
    ) -> str:
        """
        Get AI comparison of treatment options
        
        Args:
            disease_name: Disease name
            organic_options: List of organic treatments
            chemical_options: List of chemical treatments
        
        Returns:
            Treatment comparison and recommendation
        """
        
        prompt = f"""Compare treatment options for {disease_name}:

**Organic Options:**
{chr(10).join(f'- {opt}' for opt in organic_options)}

**Chemical Options:**
{chr(10).join(f'- {opt}' for opt in chemical_options)}

Provide:
1. When to use organic vs chemical (2 sentences)
2. Most effective option for severe cases (1 sentence)
3. Best option for prevention (1 sentence)

Be concise and practical."""

        try:
            response = self._call_groq_api(prompt)
            return response.strip()
        except Exception as e:
            return "Both organic and chemical treatments can be effective. Choose based on severity and personal preference."


def get_groq_analyzer(api_key: Optional[str] = None) -> Optional[GroqAnalyzer]:
    """
    Get Groq analyzer instance
    
    Args:
        api_key: Groq API key (optional, will check env if not provided)
    
    Returns:
        GroqAnalyzer instance or None if API key not available
    """
    
    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')
    
    if api_key:
        return GroqAnalyzer(api_key)
    
    return None
