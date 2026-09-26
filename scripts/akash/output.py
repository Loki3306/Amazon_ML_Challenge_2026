import os


def write_submission_tsv(filepath, mapping_dict, id_column_name, ordered_s1_ids):
    """
    Writes a tab-separated submission file.
    mapping_dict: dict of s1_id -> iterable of target_ids
    ordered_s1_ids: list of all S1 IDs in original test order
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        f.write(f'source1_entity_id\t{id_column_name}\n')
        for s1_id in ordered_s1_ids:
            target_ids = mapping_dict.get(s1_id, [])
            if isinstance(target_ids, (set, list, tuple)):
                # Deduplicate and sort for determinism
                cleaned_ids = sorted(list(set(target_ids)))
                val_str = ','.join(cleaned_ids)
            elif isinstance(target_ids, str):
                val_str = target_ids.strip()
            else:
                val_str = ''
            f.write(f'{s1_id}\t{val_str}\n')


def write_final_report(filepath, report_dict):
    """
    Writes a human-readable final summary report to output/final_report.txt.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION\n')
        f.write('FINAL PIPELINE REPORT\n')
        f.write('=' * 60 + '\n\n')

        for section, content in report_dict.items():
            f.write(f'[{section}]\n')
            if isinstance(content, dict):
                for k, v in content.items():
                    f.write(f'  {k:<30}: {v}\n')
            elif isinstance(content, list):
                for item in content:
                    f.write(f'  - {item}\n')
            else:
                f.write(f'  {content}\n')
            f.write('\n')
