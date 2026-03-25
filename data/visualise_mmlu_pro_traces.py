import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 1. Data Loading and Flattening
# ==========================================
def load_and_flatten_data(jsonl_path: str, split_name: str) -> pd.DataFrame:
    records = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            
            question_text = row.get('question', 'Unknown')
            question_category = row.get('category', 'Unknown')
            traces = row.get('traces', [])
            annotations = row.get('annotations', [])
            
            for trace, ann in zip(traces, annotations):
                reasoning_text = trace.get('reasoning_trace', '')
                trace_len = len(reasoning_text.split()) if reasoning_text else 0
                
                records.append({
                    'split': split_name,
                    'question_id': hash(question_text),
                    'category': question_category,
                    'mode': trace.get('mode'),
                    'final_answer': trace.get('final_answer'),
                    'auto_correct': trace.get('auto_correct'),
                    'confidence': trace.get('self_reported_confidence'),
                    'trace_length': trace_len,
                    'reasoning_type': ann.get('reasoning_type'),
                    'knowledge_intensity': ann.get('knowledge_intensity'),
                    'error_type': ann.get('error_type'),
                    'quality_score': ann.get('quality_score'),
                    'verification': ann.get('verification')
                })
                
    return pd.DataFrame(records)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0}) 

# ==========================================
# 2. Visualization Functions
# ==========================================

def get_save_path(base_dir: str, split: str, filename: str) -> str:
    split_dir = os.path.join(base_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, filename)

def plot_accuracy_by_mode(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(10, 6))
    
    mode_stats = df.groupby('mode')['auto_correct'].agg(['mean', 'count'])
    mode_stats['mean'] *= 100
    mode_stats = mode_stats.sort_values(by='mean', ascending=False)
    
    ax = sns.barplot(x=mode_stats['mean'].values, y=mode_stats.index, palette='viridis')
    plt.title(f'Trace-Level Accuracy by Expert Mode ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Percentage Correct (%)', fontsize=12)
    plt.ylabel('Generation Mode', fontsize=12)
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        count = mode_stats['count'].iloc[i]
        plt.text(width + 1, p.get_y() + p.get_height()/2. + 0.1, 
                 f'{width:.1f}% (N={count})', ha='left', va='center')
        
    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'accuracy_by_mode.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

# --- NEW FUNCTION: Analyzes accuracy at the QUESTION level ---
def plot_question_level_accuracy(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(10, 6))
    
    # Aggregate data by question_id
    q_df = df.groupby('question_id').agg(
        total_traces=('auto_correct', 'count'),
        correct_traces=('auto_correct', 'sum')
    )
    
    # Calculate different question-level metrics
    q_df['pass_any'] = q_df['correct_traces'] > 0
    q_df['majority_vote'] = q_df['correct_traces'] >= (q_df['total_traces'] / 2)
    q_df['strict_pass'] = q_df['correct_traces'] == q_df['total_traces']
    
    metrics = {
        'Pass@Any (At least 1 trace correct)': q_df['pass_any'].mean() * 100,
        'Majority Vote (>= 50% traces correct)': q_df['majority_vote'].mean() * 100,
        'Strict Pass (All traces correct)': q_df['strict_pass'].mean() * 100
    }
    
    ax = sns.barplot(x=list(metrics.values()), y=list(metrics.keys()), palette='magma')
    plt.title(f'Question-Level Committee Performance ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Percentage of Questions (%)', fontsize=12)
    
    total_q = len(q_df)
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 1, p.get_y() + p.get_height()/2. + 0.1, 
                 f'{width:.1f}% (N={total_q} Qs)', ha='left', va='center')
        
    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'question_level_accuracy.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

# --- UPDATED FUNCTION: Shows Category Performance using Majority Vote ---
def plot_subject_performance(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(12, 6))
    
    # First, aggregate to the question level to prevent easy questions with 5 traces from skewing the mean
    q_df = df.groupby(['question_id', 'category']).agg(
        total_traces=('auto_correct', 'count'),
        correct_traces=('auto_correct', 'sum')
    ).reset_index()
    
    # Define "correct" for a question as having a majority of correct traces
    q_df['majority_correct'] = q_df['correct_traces'] >= (q_df['total_traces'] / 2)
    
    cat_stats = q_df.groupby('category')['majority_correct'].agg(['mean', 'count'])
    cat_stats['mean'] *= 100
    cat_stats = cat_stats.sort_values(by='mean', ascending=False)
    
    ax = sns.barplot(x=cat_stats.index, y=cat_stats['mean'].values, palette='crest')
    plt.title(f'Category Accuracy (Majority Vote) ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Percentage of Questions Correct (%)', fontsize=12)
    
    new_labels = [f"{label.get_text()}\n({cat_stats['count'].loc[label.get_text()]} Qs)" for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'accuracy_by_category.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_reasoning_type_vs_correctness(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(12, 6))
    
    counts = df['reasoning_type'].value_counts()
    crosstab = pd.crosstab(df['reasoning_type'], df['auto_correct'], normalize='index') * 100
    crosstab = crosstab.sort_values(by=True, ascending=False)
    
    ax = crosstab.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], figsize=(12,6))
    plt.title(f'Correctness Rate by Dominant Reasoning Type ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Reasoning Type', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(['Incorrect', 'Correct'], title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    new_labels = [f"{label.get_text()}\n(N={counts.get(label.get_text(), 0)} Traces)" for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'reasoning_type_vs_correctness.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_confidence_calibration(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(8, 6))
    
    conf_df = df.dropna(subset=['confidence']).copy()
    if conf_df.empty:
        plt.close()
        return
        
    order = ['low', 'medium', 'high']
    counts = conf_df['confidence'].value_counts()
    
    ax = sns.barplot(x='confidence', y='auto_correct', data=conf_df, order=order, palette='coolwarm')
    plt.title(f'Uncertainty Calibration: Confidence vs. Trace Accuracy ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Self-Reported Confidence', fontsize=12)
    plt.ylabel('Trace Accuracy Rate', fontsize=12)
    
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if not np.isnan(height):
            conf_level = order[i]
            count = counts.get(conf_level, 0)
            plt.text(p.get_x() + p.get_width()/2., height - 0.02, 
                     f'{height*100:.1f}%\n(N={count})', ha='center', va='top', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'confidence_calibration.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_quality_vs_correctness(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(8, 6))
    
    qual_stats = df.groupby('quality_score')['auto_correct'].agg(['mean', 'count']).reset_index()
    
    ax = sns.barplot(x='quality_score', y='mean', data=qual_stats, palette='Blues')
    plt.title(f'Annotator Quality Score vs. Trace Correctness ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Assigned Quality Score (1-5)', fontsize=12)
    plt.ylabel('Accuracy Rate', fontsize=12)
    
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if not np.isnan(height):
            count = qual_stats['count'].iloc[i]
            va, y_offset, color = ('top', -0.02, 'white') if height > 0.15 else ('bottom', 0.02, 'black')
            plt.text(p.get_x() + p.get_width()/2., height + y_offset, 
                     f'{height*100:.1f}%\n(N={count})', ha='center', va=va, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'quality_vs_correctness.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_error_type_distribution(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(10, 6))
    wrong_df = df[(df['auto_correct'] == False) & (df['error_type'] != 'none')]
    
    if wrong_df.empty:
        plt.close()
        return

    error_counts = wrong_df['error_type'].value_counts()
    ax = sns.barplot(x=error_counts.values, y=error_counts.index, palette='rocket')
    
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.5, p.get_y() + p.get_height()/2. + 0.1, f'N={int(width)} Traces', ha='left', va='center')

    plt.title(f'Distribution of Error Types ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Error Type', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'error_type_distribution.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_trace_length_distribution(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='auto_correct', y='trace_length', data=df, palette=['#e74c3c', '#2ecc71'])
    
    counts = df['auto_correct'].value_counts()
    plt.xticks([0, 1], [f'Incorrect\n(N={counts.get(False, 0)} Traces)', f'Correct\n(N={counts.get(True, 0)} Traces)'])
    
    plt.title(f'Trace Length vs. Outcome ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Was Final Answer Correct?', fontsize=12)
    plt.ylabel('Trace Length (Words)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'trace_length_distribution.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_quality_score_by_mode(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(10, 6))
    
    order = df.groupby('mode')['quality_score'].mean().sort_values(ascending=False).index
    counts = df['mode'].value_counts()
    
    sns.violinplot(x='quality_score', y='mode', data=df, order=order, 
                   inner=None, color=".9", linewidth=0, cut=0)
    
    sns.pointplot(x='quality_score', y='mode', data=df, order=order, 
                  linestyle='none', color="black", errorbar=('ci', 95), capsize=.1, markers="D")
    
    ax = plt.gca()
    new_labels = [f"{label.get_text()}\n(N={counts.get(label.get_text(), 0)} Traces)" for label in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels)
    
    plt.title(f'Quality Scores by Expert Mode: KDE + Forest Plot ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Quality Score (1-5)', fontsize=12)
    plt.ylabel('Generation Mode', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'quality_scores_forest_kde.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_inter_expert_agreement(df: pd.DataFrame, split: str, base_dir: str):
    plt.figure(figsize=(8, 6))
    
    agreement_df = df.groupby('question_id')['final_answer'].nunique().reset_index()
    agreement_df.rename(columns={'final_answer': 'unique_answers_count'}, inplace=True)
    
    counts = agreement_df['unique_answers_count'].value_counts().sort_index()
    total_questions = len(agreement_df)
    
    ax = sns.barplot(x=counts.index, y=counts.values, palette='magma')
    plt.title(f'Inter-Expert Agreement: Unique Answers per Question ({split.upper()})\nTotal Questions: {total_questions}', fontsize=14, pad=15)
    plt.xlabel('Number of Different Final Answers Proposed by Committee', fontsize=12)
    plt.ylabel('Number of Questions', fontsize=12)
    
    for p in ax.patches:
        height = p.get_height()
        plt.text(p.get_x() + p.get_width()/2., height + (height * 0.02), 
                 f'N={int(height)} Qs', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(get_save_path(base_dir, split, 'inter_expert_agreement.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================
# 3. Execution Block (Loops over splits)
# ==========================================

data_base_path = '/mnt/pdata/caf83/data/mmlu_pro_modes'
output_plot_dir = 'data/mmlu_multi_reasoning_plots'
splits = ['train', 'eval', 'test']

for split in splits:
    file_path = os.path.join(data_base_path, f'{split}.jsonl')
    
    if not os.path.exists(file_path):
        print(f"Skipping {split} - File not found: {file_path}")
        continue
        
    print(f"Processing and plotting {split} split...")
    df = load_and_flatten_data(file_path, split_name=split)
    
    if df.empty:
        print(f"Warning: No data found in {split} split.")
        continue
    
    plot_accuracy_by_mode(df, split, output_plot_dir)
    plot_question_level_accuracy(df, split, output_plot_dir)  # NEW: Plot question-level outcomes
    plot_subject_performance(df, split, output_plot_dir)      # UPDATED: Uses Majority Vote
    plot_error_type_distribution(df, split, output_plot_dir)
    plot_reasoning_type_vs_correctness(df, split, output_plot_dir)
    plot_trace_length_distribution(df, split, output_plot_dir)
    plot_quality_score_by_mode(df, split, output_plot_dir)
    plot_confidence_calibration(df, split, output_plot_dir)
    plot_quality_vs_correctness(df, split, output_plot_dir)
    plot_inter_expert_agreement(df, split, output_plot_dir)

print(f"All plots successfully generated and saved to {output_plot_dir}/")