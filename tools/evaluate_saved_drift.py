"""Re-evaluate the released drift lineage without generating accounts or calling APIs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'dynamic_drift'))
import argparse,json,shutil
from types import SimpleNamespace
import pandas as pd
import run_multiround_generalization_benchmark as benchmark
from release_support import DEFAULT_CONFIG, apply_paths

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--input-dir',type=Path,default=ROOT/'dynamic_drift/continuous_generalization_formal_v1')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--skip-bge',action='store_true')
    parser.add_argument('--graph',action='store_true',help='Also retrain graph baselines, including RGCN.')
    parser.add_argument('--graph-epochs',type=int,default=None)
    cli=parser.parse_args();source=cli.input_dir.resolve();out=cli.output_dir.resolve()
    if source==out or source in out.parents:
        parser.error('Output must be separate from the released input dataset.')
    out.mkdir(parents=True,exist_ok=True)
    apply_paths(benchmark.__dict__,'dynamic_drift')
    settings=json.loads((source/'run_config.json').read_text())['args']
    settings.update(skip_bge=cli.skip_bge,output_dir=str(out))
    if cli.graph_epochs is not None:settings['graph_epochs']=cli.graph_epochs
    args=SimpleNamespace(**settings)
    records=[]
    for row in pd.read_csv(source/'lineage_summary.csv').fillna('').to_dict('records'):
        family=str(row['family']);generation=int(row['generation'])
        selection=str(row['selection_status'])
        run_dir=source/family/f'generation_{generation:03d}_{selection}'
        if not run_dir.is_dir():raise FileNotFoundError(run_dir)
        records.append(benchmark.RunRecord(family,generation,run_dir,run_dir/'child_manifest.json',str(row['plan_id']),str(row['child_family_version']),tuple(str(row['target_axes']).split(',')),float(row['target_pass_rate']),float(row['cumulative_target_pass_rate']),int(row['severe_non_target_count']),str(row['gate_passed']).lower()=='true',selection,{},{}))
    # Check cache row identity before reusing any saved embeddings.
    generated=benchmark.collect_generated_docs(records,args)
    published=pd.read_csv(source/'generated_doc_table.csv')
    keys=['family','generation','user_id','doc']
    if not generated[keys].reset_index(drop=True).equals(published[keys].reset_index(drop=True)):
        raise ValueError('Published embedding row order does not match the selected lineage.')
    if not cli.skip_bge:
        shutil.copyfile(source/'generated_doc_embeddings.npy',out/'generated_doc_embeddings.npy')
    benchmark.evaluate_text_tfidf(records,out,args)
    benchmark.evaluate_text_bge(records,out,args)
    if cli.graph: benchmark.evaluate_graph_generalization(records,out,args)
    print(json.dumps({'records':len(records),'accounts':len(generated),'generation':False,'output':str(out)},indent=2))

if __name__=='__main__':main()
