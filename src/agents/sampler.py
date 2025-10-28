"""
Sampling agent responsible for sampling tables from data sources and generating transformation tasks.
"""
import random
import re
import pandas as pd
import numpy as np
from src.agents.base import Agent
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Callable
from src.pipeline.operations import *
from src.utils.logger import get_logger
import time
from src.pipeline.factory import OperationFactory
from src.pipeline.base import Pipeline, Operation


class SamplerAgent(Agent):
    """
    Sampling agent responsible for sampling tables from data sources and generating transformation tasks.
    
    Main functionalities:
    1. Sample tables from data sources.
    2. Randomly generate transformation operation chains.
    3. Apply transformations to generate input and target tables.
    """
    
    def __init__(self, config: Dict, data_connector: Any):
        """
        Initialize SamplerAgent.
        
        Args:
            config: Configuration dictionary.
            data_connector: Data connector.
        """
        super().__init__(config)
        self.data_connector = data_connector
        self.logger = get_logger("sampler")
        
        # Set table sampling parameters
        self.sample_tables_per_run = config.get("sample_tables_per_run", 1)
        self.table_min_rows = config.get("table_min_rows", 3)
        self.table_max_rows = config.get("table_max_rows", 50)
        self.table_min_cols = config.get("table_min_cols", 3)
        
        # Set transformation chain generation parameters
        self.default_complexity = config.get("default_complexity", 3)
        self.complexity_transforms = config.get("complexity_transforms", {
            "1": 1, # Length of transformation chain with complexity 1
            "2": 2, # Length of transformation chain with complexity 2
            "3": 3, # Length of transformation chain with complexity 3
            "4": 4, # Length of transformation chain with complexity 4
            "5": 5,  # Length of transformation chain with complexity 5
            "6": 6,  # Length of transformation chain with complexity 6
            "7": 7,  # Length of transformation chain with complexity 7
            "8": 8,  # Length of transformation chain with complexity 8
        })
        
        # Initialize and register supported transformation operations to the factory
        self._register_operations()
        self.logger.info("SamplerAgent initialization completed")
    
    def run(self, complexity: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Generate data transformation tasks.
        
        Args:
            complexity: Transformation chain complexity (1-3).
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]: 
                (Input table, target table, transformation chain)
        """
        # Set complexity
        if complexity is None:
            complexity = self.default_complexity
        
        # Record start time
        start_time = time.time()
        
        # Sample tables from data sources
        tables, join_cols, src, suitable_operation = self.data_connector.run(n=self.sample_tables_per_run)
        if not tables:
            self.logger.warning("Failed to get tables from data source")
            return None, None, [], src
        input_table = tables.copy()
        if len(input_table) > 1:
            self.logger.info(f"Successfully sampled tables, count: {len(input_table)}")
        else:
            self.logger.info(f"Successfully sampled tables, count: 1")
        
        input_table_name = []
        for idx, value in enumerate(input_table):
            var_name = f"table_{idx+1}"
            input_table_name.append(var_name)
        
        # If the table is too small, generate a larger random table
        # for table in tables:
        #     if len(table) < self.table_min_rows or len(table.columns) < self.table_min_cols:
        #         table = self._generate_random_table()
        #         self.logger.info(f"Original table is too small, generated random table, shape: {table.shape}")
        
        # Analyze table structure
        table_infos = []
        for table in tables:
            table_info = self._analyze_table(table)
            table_infos.append(table_info)
            self.logger.debug(f"Table analysis result: {len(table_info['numeric_cols'])} numeric columns, "
                            f"{len(table_info['categorical_cols'])} categorical columns, "
                            f"{len(table_info['datetime_cols'])} datetime columns")
            
        # Generate transformation chain and Pipeline
        transform_chain, pipeline = self._generate_pipeline(input_table, input_table_name, table_infos, complexity, join_cols, suitable_operation)
        if not transform_chain:
            self.logger.warning("Failed to generate a valid transformation chain")
            return input_table, input_table, [], src
        
        input_table_name, input_table, transform_chain, selected_index = self.filter_transform_chain(input_table_name, input_table, transform_chain, self.logger)
        try:
            target_table = pipeline.transform(tables.copy())
            target_table = target_table[selected_index]
            if isinstance(target_table, list):
                self.logger.info(f"Successfully generated target table, count: {len(target_table)}")
            else:
                self.logger.info(f"Successfully generated target table, shape: {target_table.shape}")
        except Exception as e:
            self.logger.error(f"Failed to generate target table using pipeline system: {str(e)}")
            # self.logger.info("Fallback to traditional transformation method")
            # target_table = self._apply_transform_chain(tables, transform_chain)
        
        # Record elapsed time
        elapsed_time = time.time() - start_time
        self.logger.info(f"Sampling task completed in: {elapsed_time:.2f} seconds, "
                        f"Transformation chain length: {len(transform_chain)}, "
                        f"Target table shape: {target_table.shape}")
        
        return input_table, target_table, transform_chain, src
    
    def filter_transform_chain(self, input_table_name, input_table, transform_chain, logger=None): 
        """
        When there is no join/union, retain a single-table operation that is actually used, rename it to table_1, 
        and rewrite the table names and indices in the transform_chain.

        Returns:
            Tuple[List[str], List[Any], List[Dict], int]
        """
        # —— New: Remove all empty rename steps ——
        filtered_chain = []
        for step in transform_chain:
            if step.get('op') == 'rename':
                # Assume the rename dictionary is stored in step['columns'], skip if it's an empty dict
                rename_map = step.get('params', {}).get('rename_map', {})
                if not rename_map:
                    if logger:
                        logger.info("Dropping empty rename step")
                    continue
            if step.get('op') == 'filter':
                filter_condition = step.get('params', {}).get('condition', None)
                if not filter_condition:
                    if logger:
                        logger.info("Dropping empty filter step")
                    continue
            filtered_chain.append(step)
        transform_chain = filtered_chain
        if len(input_table) > 1 and not any(step.get('op') in ('join', 'union') for step in transform_chain):
            if logger:
                logger.info("No join/union detected. Selecting a single-table transform chain.")

            for i, table_name in enumerate(input_table_name):
                related_chain = [step for step in transform_chain if table_name in step.get('op_tables', [])]
                if related_chain:
                    if logger:
                        logger.info(f"Selected transform_chain related to {table_name} with {len(related_chain)} steps.")

                    # Modify transform_chain: replace op_tables and table_indices
                    updated_chain = []
                    for step in related_chain:
                        new_step = step.copy()
                        new_step['op_tables'] = ['table_1']
                        new_step['table_indices'] = [0]
                        updated_chain.append(new_step)

                    return ['table_1'], [input_table[i]], updated_chain, i

            if logger:
                logger.warning("No transform_chain step matched any table. Returning original.")

        return input_table_name, input_table, transform_chain, 0


    
    def _register_operations(self):
        """
        Initialize and register supported transformation operations to OperationFactory.
        
        Register all operations to the factory instead of maintaining transformation operation instances within SamplerAgent.
        """
        # Create all supported operation classes
        operations = {
            "filter": FilterOperation,
            "groupby": GroupByOperation,
            "sort": SortOperation,
            "pivot": PivotOperation,
            "unpivot": StackOperation,
            "explode": ExplodeOperation,
            "wide_to_long":WidetolongOperation,
            "join": JoinOperation,
            "union": UnionOperation,
            "transpose": TransposeOperation,
            "dropna": DropNullsOperation,
            "deduplicate": DeduplicateOperation,
            "topk": TopKOperation,
            "select": SelectColOperation,
            "cast": CastTypeOperation,
            "rename": RenameOperation,
            
            # Add other operations...
        }
        
        # Ensure only unregistered operations are registered
        registered_ops = OperationFactory.get_available_operations()
        
        for op_name, op_class in operations.items():
            if op_name not in registered_ops:
                try:
                    OperationFactory.register(op_name, op_class)
                    self.logger.debug(f"Registered operation: {op_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to register operation {op_name}: {str(e)}")
                    
    def _analyze_table(self, table: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the structure and features of a single table.
        
        Args:
            table: Input table.      
        Returns:
            Dict[str, Any]: Table analysis results.
        """
        columns = table.columns.tolist()
        dtypes = table.dtypes
        
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        text_cols = []
        
        has_list_columns = any(
            isinstance(x, list)
            for x in table.dropna().head(5).values.flatten()
        )
        
        # New: Check for comma-separated text columns
        has_comma_separated_text = False
        for col in table.select_dtypes(include=['object', 'string']).columns:
            # First take non-null values, then look at the first 5, and finally check if there are commas in these 5
            if table[col].dropna().head(5) \
                    .astype(str) \
                    .str.contains(',', regex=False) \
                    .any():
                has_comma_separated_text = True
                break
        
        # Column type classification
        for col in columns:
            # assert isinstance(col, str), f"Unexpected column name: {col}"
            dtype = dtypes[col]
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_cols.append(col)
            elif pd.api.types.is_datetime64_dtype(dtype):
                datetime_cols.append(col)
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                try:
                    unique_ratio = table[col].nunique() / len(table)
                    if isinstance(unique_ratio, pd.Series):
                        unique_ratio = unique_ratio.mean()
                    if isinstance(unique_ratio, (int, float)):
                        if unique_ratio < 0.5:
                            categorical_cols.append(col)
                        else:
                            text_cols.append(col)
                except Exception as e:
                    # print(f"Error on column {col}: {e}")
                    continue  # Skip error columns
        
        # New: Logic for determining wide format
        is_wide_format = False
        split_patterns = [
            r' ', r'_', r'-', r'\.', r'\/', 
            r'(?<=\D)(?=\d)|(?<=\d)(?=\D)'  # Implicitly split letters and numbers
        ]
        from collections import defaultdict  # Ensure import
        suffix_map = defaultdict(set)
        try:
            for col in columns:
                segments = [col]
                for pat in split_patterns:
                    new_segments = []
                    for seg in segments:
                        parts = re.split(pat, seg)
                        new_segments.extend([p for p in parts if p])
                    segments = new_segments
                
                if len(segments) < 2:
                    continue
                
                for i in range(1, len(segments)):
                    prefix = tuple(segments[:i])
                    suffix = tuple(segments[i:])
                    suffix_map[suffix].add(prefix)
        except Exception as e:
            is_wide_format = False
        
        suffixes = list(suffix_map.keys())
        for i in range(len(suffixes)):
            for j in range(i+1, len(suffixes)):
                common_prefixes = suffix_map[suffixes[i]] & suffix_map[suffixes[j]]
                if len(common_prefixes) >= 2:
                    is_wide_format = True
                    break
            if is_wide_format:
                break
            
        can_transpose = False
        if not has_list_columns and not has_comma_separated_text:
            n_rows, n_cols = table.shape

            # Determine if column names are regular (prefix + number)
            structured_column_names = 0
            for col in table.columns[1:]:  # Exclude the first column (assumed to be a label)
                if re.search(r'[A-Za-z]+[\s_\-]?\d+', str(col)):
                    structured_column_names += 1

            # First column: column name + uniqueness check
            first_col_name = str(table.columns[0]).lower()
            first_col_numeric_count = pd.to_numeric(table.iloc[:, 0], errors='coerce').notna().sum()
            if n_rows > 0:
                first_col_unique_ratio = table.iloc[:, 0].nunique() / n_rows
            else:
                first_col_unique_ratio = 0.0
            first_col_is_unique = table.iloc[:, 0].is_unique

            # Determine if the table is suitable for transposition
            if first_col_is_unique and n_rows < 10:
                if any(key in first_col_name for key in ["model", "type", "metric", "name"]) or first_col_unique_ratio > 0.5:
                    if first_col_numeric_count <= 3:
                        can_transpose = True
                    
        can_groupby = False
        if len(table.columns) > 1 and len(table) > 1:
            can_groupby = True
            
        conversions = []
        def is_number_like(x):
            try:
                float(str(x))
                return True
            except:
                return False
        # Numeric -> String
        for col in numeric_cols:
            conversions.append((col, 'str'))

        # String -> Numeric (check if convertible)
        for col in text_cols:
            if table[col].apply(is_number_like).mean() > 0.8:
                conversions.append((col, 'float'))

        # Datetime -> String
        for col in datetime_cols:
            conversions.append((col, 'str'))

        # String -> Datetime (simple sampling check)
        for col in text_cols:
            non_null_series = table[col].dropna()
            sample = non_null_series.sample(min(2, len(non_null_series))).astype(str)
            try:
                pd.to_datetime(sample)
                conversions.append((col, 'datetime64'))
            except:
                continue

        # Integer -> Float, Float -> Integer
        for col in numeric_cols:
            if pd.api.types.is_integer_dtype(table[col]):
                conversions.append((col, 'float'))
            elif pd.api.types.is_float_dtype(table[col]):
                conversions.append((col, 'int'))

        # Categorical -> String
        for col in categorical_cols:
            conversions.append((col, 'str'))

        if not conversions:
            has_castable = False
        else:
            has_castable = True
            
        can_pivot = False
        # Check if at least one non-numeric column has fewer than 10 unique values (for columns)
        has_valid_columns = any(
            not pd.api.types.is_numeric_dtype(table[col]) and table[col].nunique() < 10
            for col in categorical_cols + text_cols
        )
        # Check if there is at least one numeric column (for values)
        has_numeric_values = any(
            pd.api.types.is_numeric_dtype(table[col])
            for col in numeric_cols
        )
        if has_numeric_values and has_valid_columns and len(table.columns) >= 3 and len(categorical_cols)>=2 and numeric_cols:
            can_pivot = True
        # if not can_pivot:
        #     print("can_pivot = False")

        return {
            "columns": columns,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "text_cols": text_cols,
            "shape": table.shape,
            "has_list_columns": has_list_columns,
            "has_comma_separated_text": has_comma_separated_text,
            "is_wide_format": is_wide_format,
            "can_transpose": can_transpose,
            "can_groupby": can_groupby,
            "castable": has_castable,
            "can_pivot": can_pivot,
        }
        
    def _get_compatible_operations(self, table: pd.DataFrame, table_info: Dict) -> List[str]:
        """
        Get operations compatible with the current table, supporting `explode` for comma-separated text.
        
        Args:
            table: Current table.
            table_info: Table analysis results.
            
        Returns:
            List[str]: List of compatible operation names.
        """
        compatible_ops = []
        available_ops = OperationFactory.get_available_operations()
        
        for op_name, op_class in available_ops.items():
            operation = OperationFactory.create(op_name, {}, validate=False)
            if not operation:
                continue
            is_compatible = True
            
            # Type compatibility check
            if hasattr(operation, 'compatible_dtypes'):
                compat_dtypes = operation.compatible_dtypes
                has_compatible_dtype = False
                
                if "numeric" in compat_dtypes and table_info["numeric_cols"]:
                    has_compatible_dtype = True
                elif "categorical" in compat_dtypes and table_info["categorical_cols"]:
                    has_compatible_dtype = True
                elif "datetime" in compat_dtypes and table_info["datetime_cols"]:
                    has_compatible_dtype = True
                elif "text" in compat_dtypes and table_info["text_cols"]:
                    has_compatible_dtype = True
                elif "mixed" in compat_dtypes:
                    has_compatible_dtype = True
                
                is_compatible = has_compatible_dtype
            
            # Special operation compatibility check
            if op_name == 'explode':
                is_compatible = table_info["has_list_columns"] or table_info["has_comma_separated_text"]
            if op_name == 'wide_to_long':
                is_compatible = is_compatible and table_info["is_wide_format"]
            if op_name == 'pivot':
                # Requires at least two columns for index and columns
                is_compatible = table_info["can_pivot"]
            if op_name == 'unpivot':
                is_compatible = len(table.columns) >= 3
            if op_name == 'transpose':
                is_compatible = table_info["can_transpose"]
            if op_name == 'groupby':
                is_compatible = table_info["can_groupby"]
            if op_name == 'cast':
                is_compatible = (is_compatible and table_info["castable"])
            if is_compatible:
                compatible_ops.append(op_name)     
        return compatible_ops
    
    def _generate_pipeline(self, tables: List[pd.DataFrame], input_table_name, table_infos: List[Dict], complexity: int, join_cols: Dict, suitable_operation: List) -> Tuple[List[Dict], Pipeline]:
        """
        Generate a transformation pipeline.
        
        Args:
            tables: List of input tables.
            input_table_name: Names of input tables.
            table_infos: Analysis results of input tables.
            complexity: Transformation chain complexity.
            join_cols: Join columns information.
            suitable_operation: List of suitable operations.
            
        Returns:
            Tuple[List[Dict], Pipeline]: Transformation chain and pipeline.
        """
        num_transforms = self.complexity_transforms[str(complexity)]
        #num_transforms = random.randint(min_ops, max_ops)
        current_tables = [df.copy() for df in tables]
        current_table_infos = [info.copy() for info in table_infos]
        current_table_names = [name for name in input_table_name]
        transform_chain = []
        pipeline = Pipeline()
        
        table_histories = {i: [] for i in range(len(current_tables))}
        history_penalty = 0.1
        suitable_boost = 10
        
        SINGLE_OP_PRIORITY = {
             # Cleaning
            'deduplicate': ['sort', 'wide_to_long','explode', 'groupby', 'pivot'],
            'dropna': ['explode','wide_to_long', 'select','groupby', 'pivot'],
            'filter': ['deduplicate','select','rename','groupby','dropna'],
            
            # Data integration
            'groupby': ['sort', 'pivot', 'unpivot', 'select', 'rename','filter', 'cast'],
            'join': ['groupby', 'pivot','filter','unpivot','deduplicate'],
            'union': ['groupby', 'pivot','deduplicate','unpivot', 'filter'],
            
             # Structural reconstruction
            'pivot': ['filter', 'groupby', 'unpivot', 'rename'],  # Do not follow with explode
            'unpivot': ['filter', 'groupby', 'cast'],          # Do not follow with explode
            'explode': ['groupby', 'filter'],                 # Do not follow with pivot/stack to prevent structural confusion
            'wide_to_long': ['filter', 'groupby', 'cast'],            # Conservative follow-up                         	 # Not recommended to follow with join/union
            'transpose': ['select','unpivot','pivot','rename',],
            
             # Auxiliary
            "rename": ['filter', 'groupby','pivot', 'unpivot','select', 'transpose'],  
            "cast":["groupby","filter","sort",'pivot','unpivot', 'transpose'],
            'sort': ['topk',  'deduplicate', 'select'],
            "topk":['explode',"rename"],
            'select':['filter','wide_to_long', 'rename','groupby','dropna','transpose'],
        }

        for i in range(num_transforms):
            used_ops = [t['op'] for t in transform_chain]
            available_ops = self._get_available_ops(current_tables, current_table_infos, join_cols)
            if not available_ops:
                break
            filtered_ops = []
            for op in available_ops:
                if op['type'] == 'single' and op['op'] == 'topk':
                    table_idx = op['table_idx']
                    if 'sort' not in table_histories[table_idx]:
                        continue  # skip this op
                filtered_ops.append(op)
            available_ops = filtered_ops
            current_table_count = len(current_tables)
            op_weights = []

            # Dynamically generate priority list (multi-table operations automatically prioritized)
            def get_priority_with_multi(last_op: str, table_idx: int) -> List[str]:
                priority = []
                # Add available multi-table operations
                if current_table_count >= 2:
                    multi_ops = [op['op'] for op in available_ops 
                                if op['type'] == 'multi']
                    priority.extend(multi_ops)
                # Add single-table operation priorities
                single_priority = SINGLE_OP_PRIORITY.get(last_op, [])
                single_ops = [op['op'] for op in available_ops 
                                if op['type'] == 'single' and op['table_idx'] == table_idx]
                if not single_priority:
                    # priority.extend(single_ops)
                    priority.extend([op for op in single_priority if op in single_ops])
                else:
                    priority.extend([op for op in single_priority if op in single_ops])
                return priority
            
            # Handle weights for each operation
            for op in available_ops:
                weight = 1.0
                if op['type'] == 'single':
                    table_idx = op['table_idx']
                    last_op = table_histories[table_idx][-1] if table_histories[table_idx] else None
                    # Get dynamic priority list
                    priority = get_priority_with_multi(last_op, table_idx)
                    valid_ops = list(set([op['op'] for op in available_ops]))
                    
                    # Calculate probability
                    m = len(priority)
                    if m == 0:
                        prob = 1.0 / len(valid_ops)
                    else:
                        # Arithmetic sequence probability distribution (multi-table operations automatically prioritized)
                        total = m * (m + 1) / 2
                        try:
                            pos = priority.index(op['op'])
                            prob = (m - pos) / total * 0.9  # 90% probability assigned to priority operations
                        except ValueError:
                            prob = 0.1 / (len(valid_ops) - m)  # Remaining 10% assigned to others
                    if suitable_operation:
                        table_idx = op['table_idx']
                        if op['op'] in suitable_operation[table_idx]:
                            weight *= suitable_boost
                        if op['op'] == 'transpose' and op['op'] in suitable_operation[table_idx]:
                            weight *= 6
                        if op['op'] == 'wide_to_long' and op['op'].startswith(suitable_operation[table_idx]):
                            weight *= 8
                    if op['op'] == 'pivot':
                            weight *= 5
                    if op['op'] == 'rename' or op['op'] == 'select':
                        weight *= 0.5
                    weight *= prob

                # Multi-table operation base weight (automatically gets higher weight when table count ≥ 2)
                if op['type'] == 'multi':
                    weight *= 8 if current_table_count >= 2 else 0.5
                    if suitable_operation:
                        tables = op['tables']
                        if any(op['op'] in suitable_operation[idx] for idx in tables):
                            weight *= suitable_boost
                if op['op'] in used_ops:
                    weight *= history_penalty
                # Apply weight
                weight = max(0.005, weight)
                op_weights.append(weight)
            selected_op = random.choices(
                population=available_ops,
                weights=op_weights,
                k=1
            )[0]
            # if selected_op['op'] == 'pivot':
            #     print("selected_op = pivot")
            if selected_op['type'] == 'multi':
                # Handle multi-table operations
                op_name = selected_op['op']
                table_indices = selected_op['tables']
                df1, df2 = current_tables[table_indices[0]], current_tables[table_indices[1]]
                info1, info2 = current_table_infos[table_indices[0]], current_table_infos[table_indices[1]]
                table_names = [current_table_names[table_indices[0]], current_table_names[table_indices[1]]]

                # Create and execute operation
                temp_op = OperationFactory.create(op_name, {}, [df1, df2], [info1, info2], validate=False)
                if op_name == 'join':
                    params = temp_op.generate_params(join_cols)
                else:
                    params = temp_op.generate_params()
                operation = OperationFactory.create(op_name, params, [df1, df2], [info1, info2])
                if operation:
                    transformed = operation.transform()
                    if len(transformed) == 0:
                        del(operation)
                        continue
                    remaining_indices = [i for i in range(len(current_tables)) if i not in table_indices]
                    new_histories = {}
                    for new_idx, old_idx in enumerate(remaining_indices):
                        new_histories[new_idx] = table_histories[old_idx]
                    combined_history = []
                    # for old_idx in table_indices:
                    #     combined_history.extend(table_histories[old_idx])
                    combined_history.extend(op_name)
                    new_histories[len(remaining_indices)] = combined_history
                    table_histories = new_histories
                    
                    new_tables = [t for idx, t in enumerate(current_tables) if idx not in table_indices]
                    new_infos = [i for idx, i in enumerate(current_table_infos) if idx not in table_indices]
                    new_names = [j for idx, j in enumerate(current_table_names) if idx not in table_indices]
                    new_names.append(f"{table_names[0]}_{table_names[1]}_{op_name}")
                    new_tables.append(transformed)
                    new_infos.append(self._analyze_table(transformed))
                    current_tables = new_tables
                    current_table_infos = new_infos
                    current_table_names = new_names
                    pipeline.add_operation(operation,table_indices)
                    transform_chain.append({"op": op_name, "params": operation.params, "op_tables":table_names, "table_indices": table_indices})

            else:
                # Handle single-table operations
                table_idx = selected_op['table_idx']
                op_name = selected_op['op']
                current_df = current_tables[table_idx]
                current_info = current_table_infos[table_idx]

                # Create and execute operation
                temp_op = OperationFactory.create(op_name, {}, current_df, current_info, validate=False)
                params = temp_op.generate_params()
                operation = OperationFactory.create(op_name, params, current_df, current_info)
                if operation:
                    transformed = operation.transform()
                    # Update current table
                    if len(transformed) == 0:
                        del(operation)
                        continue
                    current_tables[table_idx] = transformed
                    current_table_infos[table_idx] = self._analyze_table(transformed)
                    # Record operation
                    pipeline.add_operation(operation,[table_idx])
                    transform_chain.append({"op": op_name, "params": operation.params, "op_tables":[current_table_names[table_idx]], "table_indices": [table_idx]})
                    table_idx = selected_op['table_idx']
                    table_histories[table_idx].append(selected_op['op'])

        return transform_chain, pipeline

    def _get_available_ops(self, current_tables: List[pd.DataFrame], current_infos: List[Dict], join_cols) -> List[Dict]:
        """
        Get available operations for the current tables.
        
        Args:
            current_tables: List of current tables.
            current_infos: Analysis results of current tables.
            join_cols: Join columns information.
            
        Returns:
            List[Dict]: List of available operations.
        """
        multi_table_ops = []
        if len(current_tables) >= 2:
            for i in range(len(current_tables)):
                for j in range(i+1, len(current_tables)):
                    if join_cols:
                        if self._check_join_possible(current_tables[i], current_tables[j]) or (i in join_cols.keys() and j in join_cols.keys()):
                            multi_table_ops.append({
                                'type': 'multi',
                                'op': 'join',
                                'tables': [i, j]
                            })
                    else:
                        if self._check_join_possible(current_tables[i], current_tables[j]):
                            multi_table_ops.append({
                                'type': 'multi',
                                'op': 'join',
                                'tables': [i, j]
                            })
                    # Union check
                    if self._check_union_possible(current_tables[i], current_tables[j]):
                        multi_table_ops.append({
                            'type': 'multi',
                            'op': 'union',
                            'tables': [i, j]
                        })
        single_table_ops = []
        for table_idx, (df, info) in enumerate(zip(current_tables, current_infos)):
            compatible_ops = self._get_compatible_operations(df, info)
            for op in compatible_ops:
                if op in ['filter', 'groupby', 'sort', 'pivot', 'unpivot', 'explode', 'wide_to_long','transpose', 'rename', 'dropna', 'deduplicate', 'topk', 'select', 'cast']:
                    single_table_ops.append({
                        'type': 'single',
                        'op': op,
                        'table_idx': table_idx
                    })
        
        return multi_table_ops + single_table_ops
    
    def _apply_transform_chain(self, input_table, transform_chain):
        """
        Apply the transformation chain to the input table (as a fallback).
        
        Args:
            input_table: Input table.
            transform_chain: Transformation chain.
            
        Returns:
            DataFrame: Transformed table.
        """
        result = input_table.copy()
        
        try:
            # Directly use Operation objects to perform transformations
            for transform in transform_chain:
                op_name = transform.get("op")
                params = transform.get("params", {})
                
                # Create and execute operation
                operation = OperationFactory.create(op_name, params)
                if operation:
                    result = operation.transform(result)
                else:
                    self.logger.warning(f"Unable to create operation: {op_name}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Failed to apply transformation chain: {str(e)}")
            # Return original table
            return input_table
    def _check_join_possible(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """
        Check if two tables can perform a JOIN operation.
        1. At least one common column as a join key.
        2. The data types of the common columns must match.
        """
        common_columns = set(df1.columns) & set(df2.columns)
        if not common_columns:
            return False
        for col in common_columns:
            if df1[col].dtype != df2[col].dtype:
                return False
        return True

    def _check_union_possible(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """
        Check if two tables can perform a UNION operation.
        1. Column names and order must match exactly.
        2. Table shapes must be compatible (row count is unrestricted, but column count must match).
        """
        if list(df1.columns) != list(df2.columns):
            return False

        for col in df1.columns:
            if df1[col].dtype != df2[col].dtype:
                return False
        return True
    
    def _generate_random_table(self) -> pd.DataFrame:
        """
        Generate a random table.
        
        Returns:
            pd.DataFrame: Randomly generated table.
        """
        num_rows = random.randint(self.table_min_rows, self.table_max_rows)
        num_cols = random.randint(self.table_min_cols, 8)
        
        data = {}
        
        # Generate ID column
        data['id'] = list(range(1, num_rows + 1))
        
        # Generate 2-3 numeric columns
        num_numeric = random.randint(2, 3)
        for i in range(num_numeric):
            col_name = f"numeric_{i+1}"
            data[col_name] = np.random.uniform(0, 100, num_rows).round(2)
        
        # Generate 1-2 categorical columns
        num_categorical = random.randint(1, 2)
        categories = [
            ['A', 'B', 'C', 'D', 'E'],
            ['Category1', 'Category2', 'Category3'],
            ['Red', 'Blue', 'Green', 'Yellow'],
            ['Small', 'Medium', 'Large']
        ]
        
        for i in range(num_categorical):
            col_name = f"category_{i+1}"
            category_values = random.choice(categories)
            data[col_name] = np.random.choice(category_values, num_rows)
        
        # Optionally add a date column
        if random.random() > 0.5:
            data['date'] = pd.date_range(start='2023-01-01', periods=num_rows)
        
        return pd.DataFrame(data)

def main():
    """
    Simple test function for SamplerAgent to directly test its functionalities.
    """
    import pandas as pd
    import numpy as np
    import json
    from pathlib import Path
    import sys
    import os
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("sampler_test")
    
    # Create a simple test table
    def create_test_table():
        return pd.DataFrame({
            'id': list(range(1, 11)),
            'name': ['Product' + str(i) for i in range(1, 11)],
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], 10),
            'price': np.random.uniform(10, 1000, 10).round(2),
            'stock': np.random.randint(0, 100, 10),
            'date': pd.date_range(start='2023-01-01', periods=10),
            'rating': np.random.uniform(1, 5, 10).round(1)
        })
    def create_sales_table():
    # Sales table: Contains sales records, linked to the product table via product_id.
        return pd.DataFrame({
            'id': list(range(10,20)),  # Corresponds to product table id
            'sale_date': pd.date_range(start='2023-02-01', periods=10),
            'quantity_sold': np.random.randint(1, 50, 10),
            'revenue': np.random.uniform(100, 5000, 10).round(2)
        })
    # Create a mock WebTablesConnector
    class MockConnector:
        def __init__(self, tables=None):
            self.tables = tables or [create_test_table(), create_sales_table()]
        
        def run(self, n=1, filters=None):
            return self.tables[:n]
    
    # Initialize mock connector
    mock_connector = MockConnector()
    
    # Create configuration
    config = {
        "default_complexity": 3,
        "complexity_transforms": {
            "1": 1,
            "2": 2,
            "3": 4
        },
        "debug_mode": True,
        "sample_tables_per_run": 2
    }
    
    # Initialize SamplerAgent
    sampler = SamplerAgent(config, mock_connector)
    
    # Test different complexities
    for complexity in [1, 2, 3]:
        logger.info(f"Testing complexity {complexity}:")
        input_table, target_table, transform_chain = sampler.run(complexity)
        
        # Print results
        logger.info(f"Transformation chain (length {len(transform_chain)}):")
        logger.info(json.dumps(transform_chain, ensure_ascii=False, indent=2))
        
        # logger.info(f"Input table (shape {input_table.shape}):")
        # logger.info(input_table.head(3).to_string())
        
        logger.info(f"Target table (shape {target_table.shape}):")
        logger.info(target_table.head(3).to_string())
        
        logger.info("-" * 50)

# def main():
#     """
#     Test function for SamplerAgent, iterating over all CSV files starting with wide_to_long in the data directory.
#     """
#     import pandas as pd
#     import numpy as np
#     import json
#     from pathlib import Path
#     import logging

#     # Configure logging
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s - %(name)s - %(levellevel)s - %(message)s'
#     )
#     logger = logging.getLogger("sampler_test")

#     # Get all matching CSV file paths (sorted by filename)
#     data_dir = Path('datas/ATDatas')
#     csv_files = sorted(data_dir.glob('wide_to_long*.csv'))
    
#     if not csv_files:
#         logger.error("No matching CSV files found")
#         return

#     # Create configuration (keep original configuration unchanged)
#     config = {
#         "default_complexity": 2,
#         "complexity_transforms": {
#             "1": {"min": 1, "max": 1},
#             "2": {"min": 1, "max": 2},
#             "3": {"min": 2, "max": 3}
#         },
#         "debug_mode": True
#     }

#     # Iterate over each CSV file for testing
#     for csv_file in csv_files:
#         logger.info(f"="*60)
#         logger.info(f"Starting test for file: {csv_file.name}")
#         logger.info(f"="*60)
        
#         # Read file and take the first 10 rows
#         df = pd.read_csv(csv_file).iloc[:10]  # Use iloc to ensure the first 10 rows
        
#         # Create a mock connector (independent for each file)
#         class MockConnector:
#             def __init__(self):
#                 self.table = df  # Directly use the current file's data
                
#             def run(self, n=1, filters=None):
#                 return [self.table]  # Return a list containing the current table

#         # Initialize SamplerAgent (reinitialize for each file)
#         sampler = SamplerAgent(config, MockConnector())

#         # Test different complexities (keep original test logic)
#         for complexity in [1, 2, 3]:
#             logger.info(f"Testing complexity {complexity}:")
#             input_table, target_table, transform_chain = sampler.run(complexity)
            
#             # Print results
#             logger.info(f"Transformation chain (length {len(transform_chain)}):")
#             logger.info(json.dumps(transform_chain, ensure_ascii=False, indent=2))
            
#             logger.info(f"Input table (shape {input_table.shape}):")
#             logger.info(input_table.head(3).to_string())
            
#             logger.info(f"Target table (shape {target_table.shape}):")
#             logger.info(target_table.head(3).to_string())
            
#             logger.info("-" * 50)

#         logger.info(f"File {csv_file.name} test completed\n\n")

if __name__ == "__main__":
    main()