"""
Definition of the pipeline system base class
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Union, Any
import re
import logging
import re
from collections import OrderedDict
from typing import Dict, List, Callable
import sqlite3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Operation(ABC):
    """Base class for transformation operations"""
    
    def __init__(self, params: Dict = None, table: Union[pd.DataFrame, List[pd.DataFrame]] = None, table_info: Union[Dict, List[Dict]] = None, validate: bool = True):
        """
        Initialize operation
        
        Args:
            params: Operation parameters
            validate: Whether to validate parameters immediately
        """
        self.params = params or {}
        if isinstance(table, list):
            self.table = [df.copy() for df in table] if table else []
        elif isinstance(table, pd.DataFrame):
            if not table.empty:
                self.table = table.copy()
            else:
                self.table = pd.DataFrame()
        else:
            self.table = pd.DataFrame()

        self.table_info = table_info.copy() if table_info else []
        if validate:
            self.validate_params()
    
    def validate_params(self):
        """Validate parameter validity, implemented by subclasses"""
        pass
    
    @abstractmethod
    def transform(self) -> pd.DataFrame:
        """
        Perform transformation operation
        
        Args:
            self.df: Input DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        pass
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """
        Safely get parameter value
        
        Args:
            name: Parameter name
            default: Default value
            
        Returns:
            Parameter value
        """
        return self.params.get(name, default)


class Pipeline:
    """Transformation pipeline"""
    
    def __init__(self, operations: List[Operation] = None, table_indices: List = None):
        """
        Initialize pipeline
        
        Args:
            operations: List of operations
        """
        self.operations = operations or []
        self.table_indices = table_indices or []
    
    def add_operation(self, operation: Operation, table_indices):
        """
        Add operation to the pipeline
        
        Args:
            operation: Operation instance
        """
        self.operations.append(operation)
        self.table_indices.append(table_indices)
    
    def transform(self, df: Union[pd.DataFrame, List[pd.DataFrame]]) -> pd.DataFrame:
        """
        Execute the complete transformation pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
            
        Raises:
            RuntimeError: When pipeline execution fails
        """
        result = df.copy()
        for i, operation in enumerate(self.operations):
            try:
                transformed = operation.transform()
                table_indices = self.table_indices[i]
                if len(table_indices) > 1:
                    next_result = [t for idx, t in enumerate(result) if idx not in table_indices]
                    next_result.append(transformed)
                else:
                    next_result = result.copy()
                    next_result[table_indices[0]] = transformed
                # Check result validity
                if transformed is None or transformed.empty:
                    raise ValueError(f"Step {i+1} operation {operation.__class__.__name__} resulted in empty output")
                    
                result = next_result
                
            except Exception as e:
                # Interrupt and return error
                raise RuntimeError(f"Pipeline execution error (step {i+1}): {str(e)}")
        
        return result 
    
    def to_transform_chain(self) -> List[Dict]:
        """
        Convert Pipeline to a transformation description chain
        
        Returns:
            List[Dict]: Transformation chain description
        """
        transform_chain = []
        for op in self.operations:
            # Get operation type name
            op_class_name = op.__class__.__name__.replace("Operation", "").lower()
            transform_chain.append({
                "op": op_class_name,
                "params": op.params.copy()  # Copy parameters to avoid reference issues
            })
        return transform_chain

    @classmethod
    def from_transform_chain(cls, transform_chain: List[Dict], input_table: Union[pd.DataFrame, List[pd.DataFrame]], factory=None):
        """
        Create Pipeline from a transformation description chain
        
        Args:
            transform_chain: Transformation chain description
            factory: Operation factory, uses default factory if None
        
        Returns:
            Pipeline: Pipeline object
        """
        exec_table = input_table.copy()
        from src.pipeline.factory import OperationFactory
        factory = factory or OperationFactory
        table_name_list = [f"table_{i+1}" for i in range(len(exec_table))]
        pipeline = cls()
        for transform in transform_chain:
            op_name = transform.get("op")
            params = transform.get("params", {})
            op_tables = transform.get("op_tables",[])
            table_idx = []
            for table_name in op_tables:
                if table_name in table_name_list:
                    table_idx.append(table_name_list.index(table_name))
            
            temp_table = [exec_table[idx] for idx in table_idx]
            indices = table_idx
            operation = factory.create(op_name, params, temp_table.copy() if len(temp_table)> 1 else temp_table[0].copy())
            if operation:
                pipeline.add_operation(operation, indices)
            transformed = operation.transform()
            if op_name != "join" and op_name != "union":
                idx = indices[0]
                exec_table[idx] = transformed
            else:
                exec_table = [t for idx, t in enumerate(exec_table) if idx not in indices]
                new_table_names = [t for idx, t in enumerate(table_name_list) if idx not in indices]
                new_table_names.append(f"{table_name_list[indices[0]]}_{table_name_list[indices[1]]}_{op_name}")
                table_name_list = new_table_names
                exec_table.append(transformed)
        return pipeline, exec_table
    
    @staticmethod
    def transform_chain_to_code(transform_chain: List[Dict], input_tables: List[str]) -> str:
        """
        Convert transformation chain to executable Python code and dynamically manage variable names based on input_tables

        Args:
            transform_chain: List of transformation chains, each element is a dict containing op, params, table_indices
            input_tables: Initial table name list for generating initial DataFrame variable names

        Returns:
            str: Generated Python code
        """
        code_lines: List[str] = []
        # Initialize variable name list, e.g., ['df', 'df_1', 'df_2', ...]
        table_vars: List[str] = ["df" if i == 0 else f"df_{i}" for i in range(len(input_tables))]

        def get_var(idx: int) -> str:
            return table_vars[idx]

        total = len(transform_chain)
        for idx, transform in enumerate(transform_chain):
            op = transform.get("op")
            params = transform.get("params", {})
            indices = transform.get("table_indices", [])
            src_vars = [get_var(i) for i in indices]
            # Determine if it's the last operation
            is_last = (idx == total - 1)
            # Default output variable name: use 'result' for the last operation, otherwise reuse or generate
            out_var = "result" if is_last else src_vars[0]

            try:
                if op == "filter":
                    cond = params["condition"]
                    line = f"{out_var} = {src_vars[0]}.query({repr(cond)})"

                elif op == "groupby":
                    by = params["by"]
                    agg = params["agg"]
                    line = (f"{out_var} = {src_vars[0]}.groupby({repr(by)})"
                            f".agg({repr(agg)})"
                            f".reset_index()")

                elif op == "sort":
                    line = f"{out_var} = {src_vars[0]}.sort_values(by={repr(params['by'])}, ascending={repr(params['ascending'])})"

                elif op == "pivot":
                    idx_cols = params["index"]
                    cols = params["columns"]
                    vals = params["values"]
                    func = params.get("aggfunc")
                    agg_arg = f", aggfunc={repr(func)}" if func else ""
                    line = (f"{out_var} = {src_vars[0]}.pivot_table(index={repr(idx_cols)}, columns={repr(cols)}, values={repr(vals)}{agg_arg})")

                elif op == "unpivot":
                    line = (f"{out_var} = {src_vars[0]}.melt(id_vars={repr(params.get('id_vars', []))}, "
                            f"value_vars={repr(params.get('value_vars', []))})")

                elif op == "explode":
                    column = repr(params['column'])
                    split_comma = params.get('split_comma', False)
                    
                    if split_comma:
                        # If split_comma is needed, split the string by commas first
                        line = f"{out_var} = {src_vars[0]}.apply(lambda x: x.split(',') if isinstance(x, str) else x).explode({column})"
                    else:
                        # Otherwise, directly call explode
                        line = f"{out_var} = {src_vars[0]}.explode({column})"

                elif op == "wide_to_long":
                    args = {
                        'stubnames': params['subnames'],
                        'i': params['i'],
                        'j': params['j'],
                        'sep': params['sep'],
                        'suffix': params['suffix']
                    }
                    arg_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                    line = f"{out_var} = pd.wide_to_long({src_vars[0]}, {arg_str}).reset_index()"

                elif op == "union":
                    left, right = src_vars
                    new_var = "result" if is_last else f"{left}_{right}_{op}"
                    line = f"{new_var} = pd.concat([{left}, {right}], ignore_index=True)"
                    if not is_last:
                        # Update variable list: remove old tables, add new table
                        for i in sorted(indices, reverse=True):
                            table_vars.pop(i)
                        table_vars.append(new_var)

                elif op == "join":
                    left, right = src_vars
                    lo = repr(params.get('left_on'))
                    ro = repr(params.get('right_on'))
                    how = repr(params.get('how', 'inner'))
                    suffixes = repr(tuple(params.get('suffixes', ('_x', '_y'))))
                    new_var = "result" if is_last else f"{left}_{right}_{op}"
                    line = (
                        f"{new_var} = {left}.merge({right}, left_on={lo}, right_on={ro},"
                        f" how={how}, suffixes={suffixes})"
                    )
                    if not is_last:
                        for i in sorted(indices, reverse=True):
                            table_vars.pop(i)
                        table_vars.append(new_var)

                elif op == "transpose":
                    line = f"{out_var} = {src_vars[0]}.T"

                elif op == "dropna":
                    sub = params.get('subset')
                    how = params.get('how', 'any')
                    arg = f"subset={repr(sub)}, how={repr(how)}" if sub else f"how={repr(how)}"
                    line = f"{out_var} = {src_vars[0]}.dropna({arg})"

                elif op == "deduplicate":
                    line = (
                        f"{out_var} = {src_vars[0]}.drop_duplicates(subset={repr(params.get('subset'))},"
                        f" keep={repr(params.get('keep', 'first'))})"
                    )

                elif op == "topk":
                    line = f"{out_var} = {src_vars[0]}.head({params['k']})"

                elif op == "select":
                    line = f"{out_var} = {src_vars[0]}.loc[:, {repr(params['columns'])}]"

                elif op == "cast":
                    line = f"{out_var} = {src_vars[0]}.astype({{{repr(params['column'])}: {repr(params['dtype'])}}})"

                elif op == "rename":
                    line = f"{out_var} = {src_vars[0]}.rename(columns={repr(params['rename_map'])})"

                else:
                    if params:
                        ps = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
                        line = f"{out_var} = {src_vars[0]}.{op}({ps})"
                    else:
                        line = f"{out_var} = {src_vars[0]}.{op}()"

            except Exception as e:
                line = f"# Error generating code for {op}: {e}"

            code_lines.append(line)

        return "\n".join(code_lines)  

    _SQL_KEYWORDS = {
        'SELECT','FROM','WHERE','GROUP','ORDER','BY','JOIN','ON','AS','LEFT','RIGHT','INNER','OUTER','UNION','ALL',
        'DISTINCT','COUNT','AVG','SUM','MIN','MAX','CASE','WHEN','THEN','ELSE','END','CAST','TABLE','INDEX','COLUMN',
        'DEFAULT','PRIMARY','KEY','FOREIGN','NOT','NULL','AND','OR','IN','LIKE','IS','BETWEEN'
    }

    @staticmethod
    def _quote_identifier(ident: Union[str, int, float]) -> str:
        """Add quotes to identifiers to handle spaces and special characters (SQLite specific)"""
        if not isinstance(ident, str):
            return str(ident)
        if ident.startswith('"') and ident.endswith('"'):
            return ident
        esc = ident.replace('"', '""')
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', ident) or ident.upper() in Pipeline._SQL_KEYWORDS:
            return f'"{esc}"'
        return ident

    @staticmethod
    def transform_chain_to_sql(chain: List[Dict], db_tables: List[str], task_id: str) -> str:  # noqa: C901
        """
        Convert transformation chain to SQL query
        
        Args:
            chain: Transformation chain
            db_tables: List of database table names
            task_id: Task identifier
            
        Returns:
            str: Generated SQL query
        """
        if not chain:
            return f"SELECT * FROM {Pipeline._quote_identifier(db_tables[0])}" if db_tables else '-- empty chain --'
        builder = _SQLBuilder(chain, db_tables, quote=Pipeline._quote_identifier)
        return builder.build(task_id)
    
    @staticmethod
    def execute_sql(sql: str, db_path: str):
        """
        Execute SQL query in SQLite database
        
        Args:
            sql: SQL query
            db_path: Database file path
            
        Returns:
            Execution success/failure, error message
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(sql)
            results = cursor.fetchall()

            # Close connection
            conn.close()
            
            return True, results
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            if 'conn' in locals():
                conn.close()
            return False, error_msg

# -------------------------------------------------------------------------
#                       Internal implementation details                     
# -------------------------------------------------------------------------
class _SQLBuilder:
    """Stateful helper that incrementally converts each DSL op to a CTE."""

    _SQL_OPERATORS = {
        '=', '!=', '<>', '>', '<', '>=', '<=',
        'AND', 'OR', 'NOT', 'IN', 'LIKE', 'IS', 'BETWEEN'
    }
    
    # --- maps -----------------------------------------------------------
    _SQL_FUNC_MAP = {'mean': 'AVG', 'avg': 'AVG', 'sum': 'SUM', 'min': 'MIN', 'max': 'MAX',
                     'count': 'COUNT', 'nunique': 'COUNT(DISTINCT', 'std': 'STDDEV', 'var': 'VARIANCE'}
    
    _JOIN_TYPE_MAP = {'inner': 'INNER', 'left': 'LEFT', 'right': 'LEFT', 'outer': 'LEFT', 'cross': 'CROSS'}

    _COL_TOKEN = re.compile(
        r'(?P<quote>["\']?)'                      # optional quote
        r'(?P<word>[^"\'=>!,\s]+(?:\s[^"\'=>!,\s]+)*)'  # word may contain spaces / ()
        r'(?P=quote)?',
        re.VERBOSE,
    )

    # ------------------------------------------------------------------
    def __init__(self, chain: List[Dict], db_tables: List[str], *, quote: Callable[[str], str]):
        self.chain = chain
        self.quote = quote
        
        # table_mapping: symbolic (table_1) → SQL reference (quoted name or CTE id)
        self.table_mapping = OrderedDict((f"table_{i + 1}", self.quote(name)) for i, name in enumerate(db_tables))

        self.column_mapping: Dict[str, str] = {}
        self.current_schema: set[str] = set()  # we lazily update once we know the real columns
        self.current_ref: str | None = next(iter(self.table_mapping.values()))

        self._ctes: List[str] = []

        # Register op → handler lookup table
        self._handlers: Dict[str, Callable[[Dict, int], None]] = {
            "filter": self._handle_filter,
            "where": self._handle_filter,
            "select": self._handle_select,
            "join": self._handle_join,
            "groupby": self._handle_groupby,
            "sort": self._handle_sort,
            "orderby": self._handle_sort,
            "rename": self._handle_rename,
            "union": self._handle_union,
            "cast": self._handle_cast,
            "astype": self._handle_cast,
            "dropna": self._handle_dropna,
        }

    def _quote_cols_in_cond(self, cond: str) -> str:
        """
        Quote column names in conditions
        
        Args:
            cond: Condition string
            
        Returns:
            str: Condition with quoted column names
        """
        def repl(m: re.Match[str]) -> str:
            tok = m.group(0)

            # Already quoted, SQL operator, number, function name → return as is
            if tok.startswith(('"', "'")) and tok.endswith(('"', "'")): return tok
            if tok.upper() in self._SQL_OPERATORS:                           return tok
            if re.fullmatch(r'\d+(?:\.\d+)?', tok):                    return tok
            if tok.endswith('('):                                      return tok

            physical = self.column_mapping.get(tok, tok)
            return self.quote(physical)

        return self._COL_TOKEN.sub(repl, cond)
    
    def build(self, task_id: str) -> str:
        """
        Iterate through the chain and construct final SQL text.
        
        Args:
            task_id: Task identifier
            
        Returns:
            str: Final SQL query
        """
        # task_id = None
        if task_id == '56d89ac1':
            print(self.chain)
        for idx, step in enumerate(self.chain, start=1):
            if not isinstance(step, dict):
                logger.warning("Step %s is not a dict – skipping", idx)
                continue
            op = str(step.get("op", "").lower())
            handler = self._handlers.get(op, self._handle_fallback)
            handler(step, idx)
        
        # Compose final SQL ------------------------------------------------
        if not self._ctes:
            return f"SELECT * FROM {self.current_ref}"

        with_clause = "WITH\n" + ",\n".join(self._ctes)
        final_select = f"\nSELECT * FROM {self.current_ref}"
        return with_clause + final_select

    def _add_cte(self, idx: int, sql_body: str, output_schema: set[str] = None) -> str:
        """
        Add a Common Table Expression (CTE)
        
        Args:
            idx: Step index
            sql_body: SQL body
            output_schema: Output schema
            
        Returns:
            str: CTE name
        """
        cte_name = f"cte_{idx}"
        self._ctes.append(f"{cte_name} AS (\n  {sql_body}\n)")
        self.current_ref = cte_name
        if output_schema is not None:
            self.current_schema = set(output_schema)
        return cte_name

    def _resolve_tables(self, symbolic: List[str]) -> List[str]:
        """
        Translate symbolic table names to SQL identifiers (quoted or CTE).
        
        Args:
            symbolic: List of symbolic table names
            
        Returns:
            List[str]: List of SQL identifiers
        """
        return [self.table_mapping.get(sym, sym) for sym in symbolic]

    def _rewrite_condition(self, cond: str) -> str:
        """
        Replace logical column names in a condition according to column_mapping.
        
        Args:
            cond: Condition string
            
        Returns:
            str: Rewritten condition
        """
        if not self.column_mapping:
            return cond
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.column_mapping)) + r")\b")
        return pattern.sub(lambda m: self.quote(self.column_mapping[m.group(0)]), cond)

    def _handle_filter(self, step: Dict, idx: int) -> None:
        """
        Handle filter operation
        
        Args:
            step: Transformation step
            idx: Step index
        """
        raw_cond = str(step.get("params", {}).get("condition", "")).strip()
        if not raw_cond:
            logger.warning("Filter step %s missing condition – skipped", idx)
            return
        safe_cond = self._quote_cols_in_cond(raw_cond)
        sql = f"SELECT * FROM {self.current_ref} WHERE {safe_cond}"
        self._add_cte(idx, sql)

    def _handle_select(self, step: Dict, idx: int) -> None:
        """
        Handle select operation
        
        Args:
            step: Transformation step
            idx: Step index
        """
        cols = step.get("params", {}).get("columns", [])
        
        if not cols:
            # No columns specified, retain all columns
            sql = f"SELECT * FROM {self.current_ref}"
            self._add_cte(idx, sql)
            return

        # Columns specified, retain only these columns
        quoted = ", ".join(self.quote(self.column_mapping.get(c, c)) for c in cols)
        sql = f"SELECT {quoted} FROM {self.current_ref}"
        
        # Update schema, currently only the selected columns remain
        self._add_cte(idx, sql)
        self.current_schema = set(cols)     

    def _find_required_columns(self, current_idx: int) -> set:
        """
        Find columns required by subsequent steps
        
        Args:
            current_idx: Current step index
            
        Returns:
            set: Set of required columns
        """
        required = set()
        for step in self.chain[current_idx:]:
            op = step.get("op", "")
            params = step.get("params", {})

            if op in {"groupby"}:
                by_cols = params.get("by", [])
                if not isinstance(by_cols, list):
                    by_cols = [by_cols]
                agg_cols = list(params.get("agg", {}).keys())
                required.update(by_cols + agg_cols)

            elif op in {"sort", "orderby"}:
                by_cols = params.get("by", [])
                if not isinstance(by_cols, list):
                    by_cols = [by_cols]
                required.update(by_cols)

            elif op == "rename":
                rename_map = params.get("rename_map", {})
                if rename_map.keys():
                    required.update(rename_map.keys())

            elif op == "cast":
                col = params.get("column")
                if col:
                    required.add(col)

        return required
    
    def _handle_join(self, step: Dict, idx: int) -> None:
        """
        Handle join operation
        
        Args:
            step: Transformation step
            idx: Step index
        """
        symbolic = step.get("op_tables", [])
        if len(symbolic) < 2:
            logger.warning("Join step %s lacks two tables – skipped", idx)
            return
        left, right = self._resolve_tables(symbolic[:2])
        params = step.get("params", {})
        how = params.get("how", "inner").lower()
        join_kw = self._JOIN_TYPE_MAP.get(how, "INNER") 
        if how == "right":
            # swap left and right
            left, right = right, left
            left_on, right_on = params.get('left_on'), params.get('right_on')
            params['left_on'], params['right_on'] = right_on, left_on            
        on_clause = self._derive_join_condition(left, right, params)
        sql = f"SELECT * FROM {left} {join_kw} JOIN {right} ON {on_clause}"
        self._add_cte(idx, sql, output_schema=None)

    def _derive_join_condition(self, left: str, right: str, p: Dict) -> str:
        """
        Derive join condition
        
        Args:
            left: Left table
            right: Right table
            p: Parameters
            
        Returns:
            str: Join condition
        """
        left_on, right_on = p.get('left_on'), p.get('right_on')
        if left_on and right_on:
            lcol = self.quote(self.column_mapping.get(left_on,  left_on))
            rcol = self.quote(self.column_mapping.get(right_on, right_on))
            return f'{left}.{lcol} = {right}.{rcol}'

        col = p.get('on')
        if isinstance(col, str) and not re.search(r'[<>=]', col):
            phys = self.quote(self.column_mapping.get(col, col))
            return f'{left}.{phys} = {right}.{phys}'

        return str(col or '1=1')  # fallback

    def _handle_groupby(self, step: Dict, idx: int) -> None:
        """
        Handle groupby operation
        
        Args:
            step: Transformation step
            idx: Step index
        """
        params = step.get("params", {})
        by_cols = params.get("by", [])
        aggs: Dict[str, str] = params.get("agg", {})

        by_sql = ", ".join(self.quote(self.column_mapping.get(c, c)) for c in by_cols)
        agg_parts = []
        for col, func in aggs.items():
            func_sql = self._SQL_FUNC_MAP.get(func.lower(), func.upper())
            target = self.quote(self.column_mapping.get(col, col))
            out_alias = f"{col}_{func}"
            agg_parts.append(f"{func_sql}({target}) AS {self.quote(out_alias)}")
            # **do not overwrite**: keep logical→new alias mapping while retaining old key
            self.column_mapping.setdefault(col, out_alias)
        sql = f"SELECT {by_sql}, {', '.join(agg_parts)} FROM {self.current_ref} GROUP BY {by_sql}"
        # update schema to include new aliases + by_cols
        self.current_schema = set(by_cols) | {f"{c}_{f}" for c, f in aggs.items()}
        self._add_cte(idx, sql)

    def _handle_sort(self, step: Dict, idx: int) -> None:
        """
        Handle sort operation
        
        Args:
            step: Transformation step
            idx: Step index
        """
        p = step.get("params", {})
        by, asc = p.get("by", []), p.get("ascending", True)
        if not isinstance(by, list):
            by = [by]
        if not isinstance(asc, list):
            asc = [asc] * len(by)
        order = [f"{self.quote(self.column_mapping.get(c, c))} {'ASC' if a else 'DESC'}" for c, a in zip(by, asc)]
        sql = f"SELECT * FROM {self.current_ref} ORDER BY {', '.join(order)}"
        self._add_cte(idx, sql)

    def _handle_rename(self, step: Dict, idx: int) -> None:
        """
        Handle rename operation
        
        Args:
            step: Transformation step
            idx: Step index
        """
        mapping = step.get("params", {}).get("rename_map", {})
        
        # Current schema, if not available, default to allowing all columns
        all_current_cols = self.current_schema if self.current_schema else set()
        
        # --- 1. Process rename mapping ---
        select_parts = []
        renamed_cols = set()
        
        for old, new in mapping.items():
            physical_old = self.column_mapping.get(old, old)
            select_parts.append(f"{self.quote(physical_old)} AS {self.quote(new)}")
            # Update column mapping
            self.column_mapping[old] = new
            renamed_cols.add(old)

        # --- 2. Retain other columns that are not renamed ---
        # If tracking schema, use schema; otherwise select *
        untouched_cols = (all_current_cols - renamed_cols) if all_current_cols else None

        if untouched_cols is None:
            # Schema unknown, select * and additionally select renamed columns
            sql = f"SELECT *, {', '.join(select_parts)} FROM {self.current_ref}"
            # Update schema to only record new renamed columns, other columns unknown
            self.current_schema = None
        else:
            # Schema known, list all columns precisely
            untouched_parts = [self.quote(self.column_mapping.get(col, col)) for col in untouched_cols]
            full_select = untouched_parts + select_parts
            sql = f"SELECT {', '.join(full_select)} FROM {self.current_ref}"
            # Update schema: retain unrenamed columns + new columns
            output_schema = untouched_cols | {new for _, new in mapping.items()}
            self.current_schema = output_schema  # Directly update            

        self._add_cte(idx, sql, output_schema=self.current_schema)        

    def _handle_union(self, step: Dict, idx: int) -> None:
        """
        Handle union operation
        
        Args:
            step: Transformation step
            idx: Step index
        """
        symbolic = step.get("op_tables", [])
        if len(symbolic) < 2:
            logger.warning("Union step %s lacks two tables – skipped", idx)
            return
        left, right = self._resolve_tables(symbolic[:2])
        sql = f"SELECT * FROM {left} UNION ALL SELECT * FROM {right}"
        self._add_cte(idx, sql)
    
    def _handle_cast(self, step: Dict, idx: int) -> None:
        """
        Handle cast operation
        
        Args:
            step: Transformation step
            idx: Step index
        """
        params = step.get("params", {})
        col, dtype = params.get("column"), str(params.get("dtype", "text")).lower()

        # 1) Map logical column to physical column
        physical = self.column_mapping.get(col, col)

        # 2) If tracking schema, first confirm physical column exists
        if self.current_schema is not None and physical not in self.current_schema:
            logger.warning("Cast step %s references missing column '%s' – skipped", idx, physical)
            return

        # 3) SQLite type mapping
        dtype_map = {
            "int": "INTEGER", "integer": "INTEGER", "int64": "INTEGER", "int32": "INTEGER",
            "float": "REAL", "double": "REAL", "float64": "REAL", "float32": "REAL",
            "bool": "INTEGER", "boolean": "INTEGER",
            "date": "TEXT", "datetime": "TEXT", "datetime64[ns]": "TEXT",
            "str": "TEXT", "string": "TEXT", "object": "TEXT",
        }
        sql_type = dtype_map.get(dtype, "TEXT")

        # 4) Generate alias, keep corresponding to logical column
        alias = f"{col}_cast"

        # 5) Generate SQL
        sql = (
            f"SELECT *, CAST({self.quote(physical)} AS {sql_type}) "
            f"AS {self.quote(alias)} FROM {self.current_ref}"
        )

        # 6) Update schema (None → set also needs fallback)
        output_schema = (self.current_schema or set()).copy()
        output_schema.add(alias)
        
        self._add_cte(idx, sql, output_schema=output_schema)

    def _handle_dropna(self, step: Dict, idx: int) -> None:
        """
        Handle dropna operation
        
        Args:
            step: Transformation step
            idx: Step index
        """
        params = step.get("params", {})
        subset = params.get("subset", [])
        how = params.get("how", "any")
        if not subset:
            logger.warning("dropna step %s without subset – skipped", idx)
            return
        if not isinstance(subset, list):
            subset = [subset]
        predicates = [f"{self.quote(col)} IS NOT NULL" for col in subset]
        cond = " AND ".join(predicates) if how == "any" else " OR ".join(predicates)
        sql = f"SELECT * FROM {self.current_ref} WHERE {cond}"
        self._add_cte(idx, sql)

    def _handle_fallback(self, step: Dict, idx: int) -> None:
        """
        Handle unsupported operations
        
        Args:
            step: Transformation step
            idx: Step index
        """
        op = step.get("op", "unknown")
        logger.warning("Unsupported op '%s' at step %s – passthrough", op, idx)
        sql = f"SELECT * FROM {self.current_ref}"
        self._add_cte(idx, f"-- unsupported op: {op}\n  {sql}")
