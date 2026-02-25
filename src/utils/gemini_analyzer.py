"""Google Gemini AI-powered analysis for enhanced disease insights"""

import os
import json
from typing import Dict, Optional
import requests


class GeminiAnalyzer:
    """
    Enhanced AI analysis using Google Gemini for contextual disease insights
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini Analyzer
        
        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.model = "gemini-pro"
    
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
            response = self._call_gemini_api(prompt)
            return self._parse_response(response)
        except Exception as e:
            return {
                "analysis": f"Gemini analysis unavailable: {str(e)}",
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
        """Build the prompt for Gemini API"""
        
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
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API"""
        
        url = f"{self.base_url}?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Check for errors
            if response.status_code != 200:
                error_detail = response.text
                raise Exception(f"API Error {response.status_code}: {error_detail}")
            
            result = response.json()
            
            if 'candidates' not in result or len(result['candidates']) == 0:
                raise Exception("No response from Gemini API")
            
            if 'content' not in result['candidates'][0]:
                raise Exception("Invalid response format from Gemini")
            
            text = result['candidates'][0]['content']['parts'][0]['text']
            return text
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
        except KeyError as e:
            raise Exception(f"Invalid API response format: {str(e)}")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
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
            "analysis": analysis.strip() or "Gemini analysis completed successfully.",
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
            response = self._call_gemini_api(prompt)
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
            response = self._call_gemini_api(prompt)
            return response.strip()
        except Exception as e:
            return "Both organic and chemical treatments can be effective. Choose based on severity and personal preference."


def get_gemini_analyzer(api_key: Optional[str] = None) -> Optional[GeminiAnalyzer]:
    """
    Get Gemini analyzer instance
    
    Args:
        api_key: Gemini API key (optional, will check env if not provided)
    
    Returns:
        GeminiAnalyzer instance or None if API key not available
    """
    
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if api_key:
        return GeminiAnalyzer(api_key)
    
    return None
