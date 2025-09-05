#!/usr/bin/env python3
"""
Data Preprocessing Utilities for AI Recruitment Assistant
=========================================================

Utilities for cleaning, validating, and preparing recruitment data for training.
Handles various data formats and sources including CSV, JSON, and text files.

Features:
- Data cleaning and validation
- Format conversion (CSV, JSON, Alpaca format)
- Text preprocessing and normalization
- Data quality assessment
- Dataset splitting and balancing
- Privacy-aware text scrubbing
"""

import os
import re
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import argparse
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    total_examples: int
    avg_instruction_length: float
    avg_input_length: float
    avg_output_length: float
    empty_instructions: int
    empty_outputs: int
    duplicate_examples: int
    quality_score: float

class RecruitmentDataProcessor:
    """Processes recruitment data for training."""
    
    def __init__(self):
        # Common patterns for data cleaning
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Privacy-sensitive patterns
        self.sensitive_patterns = {
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'salary_specific': re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'),  # Specific salary amounts
        }
    
    def clean_text(self, text: str, remove_emails: bool = True, remove_phones: bool = True) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = self.url_pattern.sub('[URL]', text)
        
        # Optionally remove emails and phones
        if remove_emails:
            text = self.email_pattern.sub('[EMAIL]', text)
        if remove_phones:
            text = self.phone_pattern.sub('[PHONE]', text)
        
        # Remove sensitive information
        for pattern_name, pattern in self.sensitive_patterns.items():
            text = pattern.sub(f'[{pattern_name.upper()}]', text)
        
        # Normalize quotes and dashes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[–—]', '-', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        return text.strip()
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from various file formats."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            return self._load_json(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._load_csv(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON data."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("JSON must contain a list or dict")
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV data."""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _load_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load plain text data (assumes simple Q&A format)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple parser for Q&A format
        examples = []
        sections = re.split(r'\n\s*\n', content.strip())
        
        for i, section in enumerate(sections):
            lines = section.strip().split('\n')
            if len(lines) >= 2:
                instruction = lines[0].strip()
                output = '\n'.join(lines[1:]).strip()
                examples.append({
                    'instruction': instruction,
                    'input': '',
                    'output': output
                })
        
        return examples
    
    def validate_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], DataQualityMetrics]:
        """Validate and assess data quality."""
        logger.info("Validating data quality...")
        
        valid_data = []
        empty_instructions = 0
        empty_outputs = 0
        
        instruction_lengths = []
        input_lengths = []
        output_lengths = []
        
        # Track duplicates
        seen_combinations = set()
        duplicates = 0
        
        for i, example in enumerate(data):
            # Ensure required fields exist
            if not isinstance(example, dict):
                logger.warning(f"Example {i} is not a dictionary, skipping")
                continue
            
            instruction = str(example.get('instruction', '')).strip()
            input_text = str(example.get('input', '')).strip()
            output = str(example.get('output', '')).strip()
            
            # Check for empty required fields
            if not instruction:
                empty_instructions += 1
                continue
            
            if not output:
                empty_outputs += 1
                continue
            
            # Check for duplicates
            combo_key = f"{instruction}|{input_text}|{output}"
            if combo_key in seen_combinations:
                duplicates += 1
                continue
            seen_combinations.add(combo_key)
            
            # Clean the data
            clean_instruction = self.clean_text(instruction)
            clean_input = self.clean_text(input_text)
            clean_output = self.clean_text(output)
            
            # Skip if cleaning removed too much content
            if len(clean_instruction) < 10 or len(clean_output) < 20:
                logger.warning(f"Example {i} too short after cleaning, skipping")
                continue
            
            # Track lengths
            instruction_lengths.append(len(clean_instruction))
            input_lengths.append(len(clean_input))
            output_lengths.append(len(clean_output))
            
            valid_data.append({
                'instruction': clean_instruction,
                'input': clean_input,
                'output': clean_output
            })
        
        # Calculate metrics
        metrics = DataQualityMetrics(
            total_examples=len(valid_data),
            avg_instruction_length=np.mean(instruction_lengths) if instruction_lengths else 0,
            avg_input_length=np.mean(input_lengths) if input_lengths else 0,
            avg_output_length=np.mean(output_lengths) if output_lengths else 0,
            empty_instructions=empty_instructions,
            empty_outputs=empty_outputs,
            duplicate_examples=duplicates,
            quality_score=self._calculate_quality_score(len(data), len(valid_data), duplicates)
        )
        
        logger.info(f"Data validation complete: {len(valid_data)}/{len(data)} examples valid")
        return valid_data, metrics
    
    def _calculate_quality_score(self, total: int, valid: int, duplicates: int) -> float:
        """Calculate a quality score (0-100)."""
        if total == 0:
            return 0.0
        
        validity_score = (valid / total) * 70  # 70% weight for validity
        uniqueness_score = ((total - duplicates) / total) * 30  # 30% weight for uniqueness
        
        return validity_score + uniqueness_score
    
    def balance_data(self, data: List[Dict[str, Any]], target_distribution: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Balance data across different categories."""
        # Simple keyword-based categorization for recruitment data
        categories = {
            'interview_scheduling': ['interview', 'schedule', 'meeting', 'time', 'available'],
            'application_status': ['application', 'status', 'update', 'received', 'review'],
            'job_offers': ['offer', 'congratulations', 'position', 'salary', 'selected'],
            'rejections': ['unfortunately', 'decision', 'another candidate', 'thank you for'],
            'job_descriptions': ['responsibilities', 'requirements', 'qualifications', 'role']
        }
        
        # Categorize examples
        categorized = {cat: [] for cat in categories.keys()}
        uncategorized = []
        
        for example in data:
            text = (example['instruction'] + ' ' + example.get('input', '') + ' ' + example['output']).lower()
            matched_category = None
            
            for category, keywords in categories.items():
                if any(keyword in text for keyword in keywords):
                    categorized[category].append(example)
                    matched_category = category
                    break
            
            if matched_category is None:
                uncategorized.append(example)
        
        # Report distribution
        logger.info("Data distribution by category:")
        for category, examples in categorized.items():
            logger.info(f"  {category}: {len(examples)} examples")
        logger.info(f"  uncategorized: {len(uncategorized)} examples")
        
        # If target distribution is specified, sample accordingly
        if target_distribution:
            balanced_data = []
            total_target = sum(target_distribution.values())
            
            for category, target_ratio in target_distribution.items():
                if category in categorized:
                    target_count = int(len(data) * (target_ratio / total_target))
                    category_data = categorized[category]
                    
                    if len(category_data) >= target_count:
                        balanced_data.extend(category_data[:target_count])
                    else:
                        # Upsample if needed
                        while len([ex for ex in balanced_data if ex in category_data]) < target_count:
                            balanced_data.extend(category_data[:target_count - len([ex for ex in balanced_data if ex in category_data])])
            
            return balanced_data + uncategorized  # Add uncategorized data
        
        return data  # Return original if no target specified
    
    def split_data(self, data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, random_state: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train/validation/test sets."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # First split: train + val vs test
        train_val, test = train_test_split(data, test_size=test_ratio, random_state=random_state, shuffle=True)
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(train_val, test_size=val_size, random_state=random_state, shuffle=True)
        
        logger.info(f"Data split: {len(train)} train, {len(val)} val, {len(test)} test")
        return train, val, test
    
    def convert_to_alpaca_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert data to Alpaca training format."""
        alpaca_data = []
        
        for example in data:
            alpaca_example = {
                'instruction': example['instruction'],
                'input': example.get('input', ''),
                'output': example['output']
            }
            alpaca_data.append(alpaca_example)
        
        return alpaca_data
    
    def save_data(self, data: List[Dict[str, Any]], output_path: str, format: str = 'json') -> None:
        """Save processed data to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(data)} examples to: {output_path}")
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        logger.info(f"✅ Data saved successfully")
    
    def generate_data_report(self, data: List[Dict[str, Any]], metrics: DataQualityMetrics, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive data report."""
        report_lines = [
            "# Data Processing Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Data Quality Metrics",
            f"- Total valid examples: {metrics.total_examples:,}",
            f"- Quality score: {metrics.quality_score:.1f}/100",
            f"- Empty instructions filtered: {metrics.empty_instructions:,}",
            f"- Empty outputs filtered: {metrics.empty_outputs:,}",
            f"- Duplicate examples removed: {metrics.duplicate_examples:,}",
            "",
            "## Text Statistics",
            f"- Average instruction length: {metrics.avg_instruction_length:.1f} characters",
            f"- Average input length: {metrics.avg_input_length:.1f} characters",
            f"- Average output length: {metrics.avg_output_length:.1f} characters",
        ]
        
        # Add sample examples
        if data:
            report_lines.extend([
                "",
                "## Sample Examples",
                ""
            ])
            
            for i, example in enumerate(data[:3]):
                report_lines.extend([
                    f"### Example {i + 1}",
                    f"**Instruction:** {example['instruction'][:100]}...",
                    f"**Input:** {example.get('input', 'N/A')[:50]}..." if example.get('input') else "**Input:** N/A",
                    f"**Output:** {example['output'][:150]}...",
                    ""
                ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report

def main():
    """Main function for data preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess recruitment data for training")
    parser.add_argument("input_file", type=str, help="Input data file (JSON, CSV, or TXT)")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--format", type=str, choices=['json', 'csv'], default='json', help="Output format")
    parser.add_argument("--split", action="store_true", help="Split data into train/val/test sets")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--balance", action="store_true", help="Balance data across categories")
    parser.add_argument("--report", type=str, help="Generate data report and save to file")
    parser.add_argument("--clean-only", action="store_true", help="Only clean data, don't convert to Alpaca format")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = RecruitmentDataProcessor()
    
    try:
        # Load data
        raw_data = processor.load_data(args.input_file)
        logger.info(f"Loaded {len(raw_data)} raw examples")
        
        # Validate and clean data
        clean_data, metrics = processor.validate_data(raw_data)
        
        # Balance data if requested
        if args.balance:
            clean_data = processor.balance_data(clean_data)
        
        # Convert to Alpaca format unless clean-only is specified
        if not args.clean_only:
            clean_data = processor.convert_to_alpaca_format(clean_data)
        
        # Handle output
        if args.split:
            # Split data
            train_data, val_data, test_data = processor.split_data(
                clean_data, args.train_ratio, args.val_ratio, 1 - args.train_ratio - args.val_ratio
            )
            
            # Save splits
            base_path = Path(args.output) if args.output else Path(args.input_file).with_suffix('')
            
            processor.save_data(train_data, f"{base_path}_train.{args.format}", args.format)
            processor.save_data(val_data, f"{base_path}_val.{args.format}", args.format)
            processor.save_data(test_data, f"{base_path}_test.{args.format}", args.format)
        else:
            # Save all data
            output_path = args.output if args.output else Path(args.input_file).with_suffix(f'.processed.{args.format}')
            processor.save_data(clean_data, output_path, args.format)
        
        # Generate report
        if args.report:
            report = processor.generate_data_report(clean_data, metrics, args.report)
            print("\n" + report)
        else:
            # Print basic stats
            print(f"\n✅ Processing complete:")
            print(f"  📊 Quality score: {metrics.quality_score:.1f}/100")
            print(f"  📝 Valid examples: {metrics.total_examples:,}")
            print(f"  ⚠️  Filtered out: {len(raw_data) - metrics.total_examples:,} examples")
    
    except Exception as e:
        logger.error(f"❌ Error processing data: {e}")
        raise

if __name__ == "__main__":
    main()
