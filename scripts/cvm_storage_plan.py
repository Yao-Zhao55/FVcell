"""Read-only cloud-disk preflight. Run on the user's CVM after access is configured."""
from pathlib import Path
import argparse,json,shutil,os
BASE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--data-root',required=True,help='User-authorized absolute data directory on CVM')
parser.add_argument('--asset',choices=['de_stats','pseudobulk'],default='de_stats')
args=parser.parse_args()
root=Path(args.data_root)
if not root.is_absolute() or root==Path(root.anchor):raise SystemExit('Specify a dedicated absolute data directory, not filesystem root')
parent=root
while not parent.exists() and parent!=parent.parent:parent=parent.parent
asset=json.loads((BASE/'large_asset_manifest.json').read_text(encoding='utf-8'))[args.asset]
free=shutil.disk_usage(parent).free
report={'planned_data_root':str(root),'existing_parent':str(parent),'parent_writable':os.access(parent,os.W_OK),'free_bytes':free,'asset':asset,'estimated_download_plus_working_space_bytes':int(asset['bytes']*1.3),'capacity_ok':free>=asset['bytes']*1.3,'network_and_charges_not_measured':True,'actions_performed':'read_only_no_directory_created_no_download'}
print(json.dumps(report,ensure_ascii=False,indent=2))
