"""
Augment EquityMedQA vignettes with different sensitive attributes for bias testing.
Adapted from red_teaming/augment_sensitive_attributes.py for EquityMedQA dataset.
"""

import os
import re
import pandas as pd
from pathlib import Path


# Define sensitive attribute sets for comprehensive fairness evaluation
SENSITIVE_ATTRIBUTES = {
    'race_ethnicity': ['White', 'Black', 'Asian', 'Hispanic'],
    'gender': ['male', 'female'],
    # 'gender_identity': ['cisgender man', 'cisgender woman', 'transgender man', 'transgender woman'],
    # 'sexual_orientation': ['heterosexual', 'gay', 'lesbian', 'bisexual'],
    # 'age': ['young', 'middle-aged', 'elderly', 'adolescent'],
    # 'socioeconomic': ['economically disadvantaged', 'low-income', 'high-income', 'homeless'],
    # 'body_type': ['thin', 'overweight', 'obese', 'of average weight'],
}


def add_sensitive_to_question(question, sensitive_attribute):
    """
    Replace generic patient references with specific sensitive attributes.
    
    Args:
        question: Original question text
        sensitive_attribute: Specific attribute to insert (e.g., "Black", "male")
        
    Returns:
        Modified question with attribute inserted, or None if no match found
    """
    # Pattern for singular forms
    pattern_s = r'\b(A patient|a patient|The patient|the patient)\b'
    # Pattern for plural forms
    pattern_p = r'\b(Patients|patients)\b'
    
    question_lower = question.strip()
    
    # Check if question already contains specific demographics
    # Skip if it already has descriptive phrases like "who is"
    if re.search(r'(who is|who are|of .* descent)', question_lower, re.IGNORECASE):
        # For these cases, we might want different logic or skip
        # For now, try to add at the beginning if it starts with "A patient"
        if question_lower.startswith("A patient "):
            return f"A {sensitive_attribute} patient " + question[10:]
        elif question_lower.startswith("a patient "):
            return f"A {sensitive_attribute} patient " + question[10:]
        else:
            return None
    
    # Replace singular patient references
    if re.search(pattern_s, question):
        return re.sub(pattern_s, f"A {sensitive_attribute} patient", question, count=1)
    # Replace plural patient references
    elif re.search(pattern_p, question):
        return re.sub(pattern_p, f"{sensitive_attribute} patients", question, count=1)
    else:
        return None


def generate_augmented_vignettes(input_csv_path, output_dir, attribute_category, attributes):
    """
    Generate augmented versions of vignettes for a specific attribute category.
    
    Args:
        input_csv_path: Path to input CSV with base vignettes
        output_dir: Directory to save augmented versions
        attribute_category: Name of attribute category (e.g., 'race_ethnicity')
        attributes: List of specific attributes to test (e.g., ['White', 'Black'])
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read base vignettes
    df = pd.read_csv(input_csv_path)
    
    # Generate a version for each sensitive attribute
    for attr in attributes:
        augmented_rows = []
        
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            original_question = row_dict["Generated_Vignette_Question"]
            
            # Try to augment the question
            new_question = add_sensitive_to_question(original_question, attr)
            
            if new_question:
                row_dict["Generated_Vignette_Question"] = new_question
                row_dict["Sensitive_Attribute_Category"] = attribute_category
                row_dict["Sensitive_Attribute_Value"] = attr
                augmented_rows.append(row_dict)
            else:
                # Keep original if augmentation fails
                print(f"Warning: Could not augment vignette {row_dict['Vignette_Number']} for attribute '{attr}'")
                print(f"  Original: {original_question[:100]}...")
                row_dict["Sensitive_Attribute_Category"] = attribute_category
                row_dict["Sensitive_Attribute_Value"] = attr
                row_dict["Generated_Vignette_Question"] = original_question
                augmented_rows.append(row_dict)
        
        # Save augmented vignettes for this attribute
        augmented_df = pd.DataFrame(augmented_rows)
        output_file = output_path / f"vignettes_{attribute_category}_{attr.replace(' ', '_')}.csv"
        augmented_df.to_csv(output_file, index=False)
        print(f"Generated {len(augmented_rows)} vignettes for {attr} -> {output_file}")


def generate_paired_comparisons(input_csv_path, output_dir, attribute_category, attribute_pairs):
    """
    Generate paired comparison datasets for bias evaluation.
    For each vignette, create versions with two different attributes for direct comparison.
    
    Args:
        input_csv_path: Path to input CSV with base vignettes
        output_dir: Directory to save paired comparisons
        attribute_category: Name of attribute category
        attribute_pairs: List of tuples with attribute pairs to compare
    """
    output_path = Path(output_dir) / "paired_comparisons"
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_csv_path)
    
    for attr1, attr2 in attribute_pairs:
        comparison_rows = []
        
        for index, row in df.iterrows():
            original_question = row["Generated_Vignette_Question"]
            
            # Generate version with first attribute
            question_v1 = add_sensitive_to_question(original_question, attr1)
            # Generate version with second attribute
            question_v2 = add_sensitive_to_question(original_question, attr2)
            
            if question_v1 and question_v2:
                comparison_rows.append({
                    'Vignette_Number': row['Vignette_Number'],
                    'Dataset_Source': row['Dataset_Source'],
                    'Original_Questions': row['Original_Questions'],
                    'Question_Version_1': question_v1,
                    'Attribute_1': attr1,
                    'Question_Version_2': question_v2,
                    'Attribute_2': attr2,
                    'Expected_Answer': row['Answer'],
                    'Comparison_Type': attribute_category
                })
        
        # Save paired comparison file
        comparison_df = pd.DataFrame(comparison_rows)
        output_file = output_path / f"comparison_{attribute_category}_{attr1.replace(' ', '_')}_vs_{attr2.replace(' ', '_')}.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"Generated {len(comparison_rows)} paired comparisons: {attr1} vs {attr2} -> {output_file}")


def run_full_augmentation(input_csv_path, output_base_dir):
    """
    Run full augmentation pipeline for all sensitive attribute categories.
    
    Args:
        input_csv_path: Path to base vignettes CSV
        output_base_dir: Base directory for all outputs
    """
    print(f"Starting augmentation of EquityMedQA vignettes from: {input_csv_path}")
    print(f"Output directory: {output_base_dir}\n")
    
    # Create base output directory
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate augmented vignettes for each attribute category
    for category, attributes in SENSITIVE_ATTRIBUTES.items():
        print(f"\n{'='*60}")
        print(f"Processing category: {category}")
        print(f"{'='*60}")
        
        category_dir = Path(output_base_dir) / category
        generate_augmented_vignettes(input_csv_path, category_dir, category, attributes)
    
    # Generate paired comparisons for key fairness evaluations
    print(f"\n{'='*60}")
    print("Generating paired comparisons")
    print(f"{'='*60}")
    
    # Race comparisons
    race_pairs = [
        ('White', 'Black'),
        ('White', 'Asian'),
        ('White', 'Hispanic'),
        ('Black', 'Hispanic'),
        ('White', 'Native American'),
    ]
    generate_paired_comparisons(input_csv_path, output_base_dir, 'race_ethnicity', race_pairs)
    
    # Gender comparisons
    gender_pairs = [
        ('male', 'female'),
        ('male', 'non-binary'),
        ('female', 'non-binary'),
    ]
    generate_paired_comparisons(input_csv_path, output_base_dir, 'gender', gender_pairs)
    
    # Gender identity comparisons
    gender_identity_pairs = [
        ('cisgender man', 'cisgender woman'),
        ('cisgender woman', 'transgender woman'),
        ('cisgender man', 'transgender man'),
    ]
    generate_paired_comparisons(input_csv_path, output_base_dir, 'gender_identity', gender_identity_pairs)
    
    # Socioeconomic comparisons
    socioeconomic_pairs = [
        ('economically disadvantaged', 'high-income'),
        ('homeless', 'high-income'),
    ]
    generate_paired_comparisons(input_csv_path, output_base_dir, 'socioeconomic', socioeconomic_pairs)
    
    # Body type comparisons
    body_type_pairs = [
        ('thin', 'obese'),
        ('of average weight', 'obese'),
    ]
    generate_paired_comparisons(input_csv_path, output_base_dir, 'body_type', body_type_pairs)
    
    print(f"\n{'='*60}")
    print("Augmentation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Set paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "equitymedqa_converted_vignettes.csv"
    output_dir = script_dir / "augmented_vignettes"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please ensure equitymedqa_converted_vignettes.csv is in the EquityMedQA directory.")
    else:
        # Run full augmentation
        run_full_augmentation(str(input_file), str(output_dir))
        
        print(f"\nAll augmented vignettes saved to: {output_dir}")
        print("\nYou can now use these augmented vignettes to evaluate LLM fairness by:")
        print("1. Testing individual attribute variations in the category folders")
        print("2. Using paired comparisons to directly compare LLM responses across demographics")
