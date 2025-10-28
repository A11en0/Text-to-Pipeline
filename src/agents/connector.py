from collections import deque
from io import StringIO
import json
import os
import random
import re
import sqlite3
import pandas as pd
from pathlib import Path
from src.agents.base import Agent
from src.utils.logger import get_logger
from multiprocessing import Manager


class WebTablesConnector(Agent):
    """
    WebTables data connector, extended to support TableBench, OpenData_CAN, and improved ATDatas prefix diversity sampling.
    Uses multiprocessing.Manager to share queue states, ensuring no duplication in a multi-process environment.
    Returns:
        tables (List[pd.DataFrame])
        join_cols (Optional[dict or None])
        src (str): Data source name
        ops (List[str] or None): Operation type corresponding to each table (only ATDatas has values)
    """
    DATA_SOURCES = ["ATDatas", "auto_pipeline", "bird", "spider_data", "TableBench", "OpenData_CAN"]
    SOURCE_WEIGHTS = [0.40,      0.20,            0.05,    0.15,             0.05,          0.15]

    def __init__(self, config):
        super().__init__(config)
        self.base_path = Path(config.get("source_path", "datas/source"))
        self.sample_size = config.get("sample_size", 1)
        self.file_types = config.get("file_types", ["csv", "xlsx", "json"])
        self.logger = get_logger("connector")
        if not self.base_path.exists():
            raise FileNotFoundError(f"Data root path does not exist: {self.base_path}")

        mgr = Manager()
        self._queues = mgr.dict()
        at_path = self.base_path / 'ATDatas'
        at_files = list(at_path.glob('**/*.csv'))
        prefix_map = {}
        for fp in at_files:
            prefix = fp.stem.split('_', 1)[0]
            prefix_map.setdefault(prefix, []).append(fp)
        file_dict = mgr.dict()
        for prefix, flist in prefix_map.items():
            random.shuffle(flist)
            file_dict[prefix] = mgr.list(deque(flist))
        prefixes = list(prefix_map.keys())
        random.shuffle(prefixes)
        self._queues['ATDatas_prefixes'] = mgr.list(prefixes)
        self._queues['ATDatas_files'] = file_dict


        ap_path = self.base_path / 'auto_pipeline'
        ap_list = []
        for folder in ap_path.iterdir():
            if folder.is_dir() and re.fullmatch(r'length(?:[0-5]|6|9)_(?:\d{1,2}|100)', folder.name):
                ap_list.extend(folder.glob('test*.csv'))
        random.shuffle(ap_list)
        self._queues['auto_pipeline'] = mgr.list(ap_list)


        bench_path = self.base_path / 'TableBench' / 'TableBench.jsonl'
        if not bench_path.exists():
            raise FileNotFoundError(f"TableBench not exists: {bench_path}")
        with open(bench_path, 'r', encoding='utf-8') as f:
            items = [json.loads(line) for line in f if line.strip()]
        random.shuffle(items)
        self._queues['TableBench'] = mgr.list(items)


        can_path = self.base_path / 'OpenData_CAN'
        gt_path = can_path / 'ground_truth.csv'
        if not gt_path.exists():
            raise FileNotFoundError(f"OpenData_CAN ground_truth.csv: {gt_path}")
        gt_df = pd.read_csv(gt_path)
        pairs = list(zip(gt_df['query_table'], gt_df['candidate_table']))
        
        random.shuffle(pairs)
        self._queues['OpenData_CAN'] = mgr.list(pairs)

        self.logger.info(f"Data source initialization completed: {self.DATA_SOURCES}")

    def run(self, n=1, filters=None):
        return self.sample_tables(n, filters)

    def sample_tables(self, n=1, filters=None):
        src = random.choices(self.DATA_SOURCES, weights=self.SOURCE_WEIGHTS, k=1)[0]
        self.logger.info(f"选择数据源: {src}")
        join_cols = None
        ops = None


        if src == 'ATDatas':
            prefixes = self._sample_from_queue('ATDatas_prefixes', 1)
            # prefixes = ['transpose']
            tables, ops = [], []
            for prefix in prefixes:
                flist = self._queues['ATDatas_files'][prefix]
                fp = flist.pop(0)  
                tables.append(pd.read_csv(fp))
                ops.append(prefix)
                flist.append(fp)  
                self._queues['ATDatas_prefixes'].append(prefix)
                
            if not tables:
                self.logger.error("ATDatas Sampling and still not getting any tables, check the file path or prefix naming!")
            return tables, join_cols, src, ops


        if src == 'auto_pipeline':
            paths = self._sample_from_queue('auto_pipeline', n)
            tables = [pd.read_csv(fp, index_col=0) for fp in paths]
            return tables, join_cols, src, ops


        if src == 'TableBench':
            items = self._sample_from_queue('TableBench', n)
            tables = []
            for it in items:
                df = pd.DataFrame(it['table']['data'], columns=it['table']['columns'])
                df = df.iloc[:, :10]
                tables.append(df)
            return tables, join_cols, src, ops

        # OpenData_CAN
        if src == 'OpenData_CAN':
            pairs = self._sample_from_queue('OpenData_CAN', 1)
            tables = []
            ops = ["union"] * n
            for qf, cf in pairs:
                qpath = self.base_path / 'OpenData_CAN' / 'tables' / qf
                cpath = self.base_path / 'OpenData_CAN' / 'tables' / cf
                qdf = pd.read_csv(qpath)
                cdf = pd.read_csv(cpath)
                
                qdf = qdf.iloc[:, :10]
                cdf = cdf.iloc[:, :10]
                
                tables.extend([qdf, cdf])
            return tables, join_cols, src, ops

        # bird 和 spider_data
        if src in ('bird', 'spider_data'):
            dfs, join_cols = self.sample_tables_and_extract_join(self.base_path / src, n)
            if src == 'bird':
                dfs = [df.sort_values(by=join_cols.get(i)).head(100) if join_cols.get(i) in df.columns else df.head(100)
                       for i, df in enumerate(dfs)]
            for i, df in enumerate(dfs):
                if df.shape[1] > 10:
                    join_col_name = join_cols.get(i)
                    columns_to_keep = [join_col_name] + [col for col in df.columns if col not in join_col_name]
                    dfs[i] = df[columns_to_keep].iloc[:, :10] 
            return dfs, join_cols, src, ops

    def _sample_from_queue(self, src, n):
        queue = self._queues[src]
        sampled = []
        for _ in range(n):
            if not queue:
                if src == 'ATDatas_prefixes':
                    prefixes = list(self._queues['ATDatas_files'].keys())
                    random.shuffle(prefixes)
                    queue.extend(prefixes)
                elif src == 'auto_pipeline':
                    ap_path = self.base_path / 'auto_pipeline'
                    files = []
                    for folder in ap_path.iterdir():
                        if folder.is_dir() and re.fullmatch(r'length(?:[0-5]|6|9)_(?:\d{1,2}|100)', folder.name):
                            files.extend(folder.glob('test*.csv'))
                    random.shuffle(files)
                    queue.extend(files)
                elif src == 'TableBench':
                    path = self.base_path / 'TableBench' / 'TableBench.jsonl'
                    items = [json.loads(line) for line in open(path, 'r', encoding='utf-8') if line.strip()]
                    random.shuffle(items)
                    queue.extend(items)
                elif src == 'OpenData_CAN':
                    gt = pd.read_csv(self.base_path / 'OpenData_CAN' / 'ground_truth.csv')
                    pairs = list(zip(gt['query_table'], gt['candidate_table']))
                    random.shuffle(pairs)
                    queue.extend(pairs)
            sampled.append(queue.pop(0))
        return sampled

    def _read_file(self, file_path):
        suffix = file_path.suffix.lower()
        if suffix == '.csv':
            return pd.read_csv(file_path)
        if suffix == '.xlsx':
            return pd.read_excel(file_path)
        if suffix == '.json':
            return pd.read_json(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")
        
    def sample_tables_and_extract_join(self, source_path: Path, n):
        """
        Randomly select a database from Spider's tables.json, sample based on column_names_original and table_names_original,
        then read table data, and finally map DataFrame column names and join_columns from original names to cleaned names.

        Parameters:
            source_path (Path): Root directory path containing tables.json and database folder.
            n (int): Number of tables to sample.

        Returns:
            dict {
                "tables": List[pd.DataFrame],       # Renamed DataFrame corresponding to each sampled table
                "join_columns": Dict[int, str]      # Dictionary of joinable column names, using cleaned names
            }
        """
        # Locate metadata and database root directory
        if source_path.name == 'spider_data':
            tables_json_path = source_path / 'tables.json'
            database_root = source_path / 'database'
        elif source_path.name == 'bird':
            tables_json_path = source_path / 'train_tables.json'
            database_root = source_path / 'train_databases_new'

        # Load metadata and randomly select a database
        with open(tables_json_path, 'r', encoding='utf-8') as f:
            tables_meta = json.load(f)
        db_meta = random.choice(tables_meta)
        db_id = db_meta['db_id']

        table_names_orig = db_meta['table_names_original']  # Original table names
        table_names_clean = db_meta['table_names']          # Cleaned table names
        num_tables = len(table_names_orig)

        # Column metadata
        cols_orig = db_meta['column_names_original']  # list of [table_idx, orig_col]
        cols_clean = db_meta['column_names']         # list of [table_idx, clean_col]

        # Build foreign key join graph (based on table indices)
        join_graph = {i: set() for i in range(num_tables)}
        for col_i, col_j in db_meta.get('foreign_keys', []):
            tbl_i, _ = cols_orig[col_i]
            tbl_j, _ = cols_orig[col_j]
            join_graph[tbl_i].add(tbl_j)
            join_graph[tbl_j].add(tbl_i)

        # Discover joinable components
        visited = set()
        components = []
        for i in range(num_tables):
            if i not in visited:
                stack, comp = [i], set()
                while stack:
                    node = stack.pop()
                    if node not in comp:
                        comp.add(node)
                        visited.add(node)
                        stack.extend(join_graph[node] - comp)
                if len(comp) >= 2:
                    components.append(comp)

        sampled_indices = None
        # Prioritize finding chains of length n in components
        for comp in components:
            if len(comp) >= n:
                paths = []
                def dfs(path, seen):
                    if len(path) == n:
                        paths.append(path.copy())
                        return
                    for neigh in join_graph[path[-1]]:
                        if neigh in comp and neigh not in seen:
                            seen.add(neigh); path.append(neigh)
                            dfs(path, seen)
                            path.pop(); seen.remove(neigh)
                for start in comp:
                    dfs([start], {start})
                    if paths:
                        break
                if paths:
                    sampled_indices = random.choice(paths)
                    sampled_indices.sort()
                    break

        # If not found, sample the largest component or all tables
        if sampled_indices is None:
            if components:
                largest = max(components, key=len)
                if len(largest) >= n:
                    sampled_indices = sorted(random.sample(largest, n))
                else:
                    sampled_indices = sorted(largest)
            else:
                sampled_indices = sorted(random.sample(range(num_tables), min(n, num_tables)))

        # Sample corresponding original table names
        sampled_tables_orig = [table_names_orig[i] for i in sampled_indices]

        # Read and rename columns
        sqlite_path = database_root / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(str(sqlite_path))
        dfs = []
        # Build column name mapping for each table: orig -> clean
        rename_maps = {}
        for meta_orig, meta_clean in zip(cols_orig, cols_clean):
            tbl_idx_o, col_o = meta_orig
            tbl_idx_c, col_c = meta_clean
            if tbl_idx_o == tbl_idx_c:
                rename_maps.setdefault(tbl_idx_o, {})[col_o] = col_c

        # Read tables and apply renaming
        for tbl_idx in sampled_indices:
            tbl_name = sampled_tables_orig[sampled_indices.index(tbl_idx)]
            df = pd.read_sql_query(f"SELECT * FROM '{tbl_name}'", conn)
            df.rename(columns=rename_maps.get(tbl_idx, {}), inplace=True)
            dfs.append(df)
        conn.close()

        # Extract join_columns using cleaned names
        join_cols = {}
        for col_i, col_j in db_meta.get('foreign_keys', []):
            tbl_i, col_o_i = cols_orig[col_i]
            tbl_j, col_o_j = cols_orig[col_j]
            if tbl_i in sampled_indices and tbl_j in sampled_indices:
                pos_i = sampled_indices.index(tbl_i)
                pos_j = sampled_indices.index(tbl_j)
                # Get clean names
                col_c_i = dict(rename_maps[tbl_i])[col_o_i]
                col_c_j = dict(rename_maps[tbl_j])[col_o_j]
                join_cols[pos_i] = col_c_i
                join_cols[pos_j] = col_c_j

        return dfs, join_cols