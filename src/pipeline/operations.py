"""
Implementation of transformation operations
"""
from collections import defaultdict
import json
import pandas as pd
import numpy as np
import random
import ast
import re
from typing import Dict, List, Optional, Any, Union
from src.pipeline.base import Operation
from src.pipeline.factory import OperationFactory


class FilterOperation(Operation):
    """Filter operation"""
    # Add compatible data types attribute
    compatible_dtypes = ["numeric", "categorical", "text"]

    def validate_params(self):
        """Validate parameter validity"""
        condition = self.get_param("condition", "")
        if condition is not None and not isinstance(condition, str):
            raise ValueError("The 'condition' parameter must be a string or None")

    def transform(self):
        df = self.table.copy()
        condition = self.get_param("condition", "")
        # If there is no condition or the condition is empty, return the original table directly
        if not condition or not isinstance(condition, str):
            return df
        condition_norm = condition

        try:
            # Try native query
            return df.query(condition_norm)
        except Exception:
            # Regex parsing: column name, operator, and value (allowing line breaks)
            pattern = r"^\s*(?P<col>.+?)\s*(?P<op>==|!=|>=|<=|>|<)\s*(?P<val>.+?)\s*$"
            m = re.match(pattern, condition_norm, flags=re.DOTALL)
            if not m:
                raise ValueError(f"Unable to parse condition: {condition}")

            col = m.group("col").strip()
            op  = m.group("op")
            val = m.group("val").strip()

            # Remove backticks
            if col.startswith("`") and col.endswith("`"):
                col = col[1:-1].replace("\\`", "`")

            # Remove outer quotes from the string
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]

            # Convert to float, if failed, keep the original string
            try:
                val_cast = float(val)
            except ValueError:
                val_cast = val

            # Construct mask based on operator
            if   op == "==": mask = df[col] == val_cast
            elif op == "!=": mask = df[col] != val_cast
            elif op == ">=": mask = df[col] >= val_cast
            elif op == "<=": mask = df[col] <= val_cast
            elif op == ">":  mask = df[col] >  val_cast
            elif op == "<":  mask = df[col] <  val_cast
            else:
                raise ValueError(f"Unsupported operator in condition: {condition}")

            return df[mask]

    def generate_params(self) -> Dict[str, Any]:
        df = self.table
        table_info = self.table_info
        if df.empty or not df.columns.tolist():
            return {}

        def quote(col: str) -> str:
            safe = col.replace("`", "\\`")
            return f"`{safe}`"

        # Prefer numeric, otherwise categorical/text
        numeric_cols = [c for c in table_info.get("numeric_cols", []) if c in df.columns]
        cat_text_cols = [c for c in (table_info.get("categorical_cols", []) + table_info.get("text_cols", [])) if c in df.columns]

        condition = None

        # Numeric column condition
        if numeric_cols:
            column = random.choice(numeric_cols)
            op = random.choice(["==", ">", "<", ">=", "<=", "!="])
            try:
                vals = df[column].dropna().unique().tolist()
                vals = [v for v in vals if isinstance(v, (int, float))]
                if vals:
                    value = random.choice(vals)
                    condition = f"{quote(column)} {op} {value}"
            except Exception:
                pass
        # String/categorical condition
        elif cat_text_cols:
            column = random.choice(cat_text_cols)
            op = random.choice(["==", "!="])
            try:
                vals = df[column].dropna().unique().tolist()
                if vals:
                    value = random.choice(vals)
                    val_repr = f"'{value}'" if isinstance(value, str) else str(value)
                    condition = f"{quote(column)} {op} {val_repr}"
            except Exception:
                pass

        # If no condition is generated, do not return condition
        if not condition:
            return {}
        return {"condition": condition}




class GroupByOperation(Operation):
    """Group by operation"""
    
    compatible_dtypes = ["categorical", "datetime", "mixed"]
    
    def validate_params(self):
        """Validate parameter validity"""
        by_cols = self.get_param("by", [])
        agg_dict = self.get_param("agg", {})
        
        if not by_cols:
            raise ValueError("Must specify grouping columns (by)")
        
        if not agg_dict:
            raise ValueError("Must specify aggregation functions (agg)")
    
    def transform(self) -> pd.DataFrame:
        """Execute group by transformation"""
        df = self.table.copy()
        by_cols = self.get_param("by", [])
        agg_dict = self.get_param("agg", {})
        
        try:
            result = df.groupby(by_cols, as_index=False).agg(agg_dict)
            return result
        except Exception as e:
            raise ValueError(f"Group by transformation failed: {str(e)}")  # Debug information
            # If it fails, try a simpler grouping
            # try:
            #     # Group by the first column only, and calculate the mean for the first numeric column
            #     by_col = by_cols[0]
            #     agg_col = list(agg_dict.keys())[0]
            #     result = df.groupby(by_col, as_index=False)[agg_col].mean()
            #     return result
            # except:
            #     # If it fails again, return the original table
            #     return df

    def generate_params(self) -> Dict[str, Any]:
        """Generate group by parameters"""
        # Select grouping columns
        df = self.table
        table_info = self.table_info
        by_cols = []
        if table_info["categorical_cols"]:
            # Prefer categorical columns
            by_cols = [random.choice(table_info["categorical_cols"])]
        elif table_info["datetime_cols"]:
            # If there are datetime columns, they can also be used
            by_cols = [random.choice(table_info["datetime_cols"])]
        elif df.columns.tolist():
            # If there are no suitable columns, randomly select a non-numeric column
            non_numeric = [col for col in df.columns if col not in table_info["numeric_cols"]]
            if non_numeric:
                by_cols = [random.choice(non_numeric)]
            else:
                by_cols = [df.columns[0]]  # If there is no other option, use the first column
        
        # Select aggregation columns and functions
        agg_dict = {}
        if table_info["numeric_cols"]:
            # Select an aggregation function for each numeric column
            for col in random.sample(table_info["numeric_cols"], 
                                    min(2, len(table_info["numeric_cols"]))):
                agg_dict[col] = random.choice(["mean", "sum", "max", "min", "count"])
        else:
            # If there are no numeric columns, use count
            if df.columns.tolist():
                col = random.choice([c for c in df.columns if c not in by_cols])
                agg_dict[col] = "count"
        
        return {"by": by_cols, "agg": agg_dict}


class SortOperation(Operation):
    """Sort operation"""
    
    compatible_dtypes = ["numeric", "categorical", "datetime", "mixed"]
    
    def validate_params(self):
        """Validate parameter validity"""
        by_cols = self.get_param("by", [])
        if not by_cols:
            raise ValueError("Must specify sorting columns (by)")
    
    def transform(self) -> pd.DataFrame:
        """Execute sort transformation"""
        df = self.table.copy()
        by_cols = self.get_param("by", [])
        ascending = self.get_param("ascending", [True] * len(by_cols))
        
        # If the length of ascending does not match, use the default value
        if len(ascending) != len(by_cols):
            ascending = [True] * len(by_cols)
        
        try:
            return df.sort_values(by=by_cols, ascending=ascending)
        except Exception as e:
            # Try sorting by the first column only
            raise ValueError(f"Sort transformation failed: {str(e)}")  # Debug information
            # try:
            #     return df.sort_values(by=by_cols[0], ascending=ascending[0])
            # except:
            #     return df

    def generate_params(self) -> Dict[str, Any]:
        """Generate sort parameters"""
        df = self.table
        table_info = self.table_info
        if not df.columns.tolist():
            return {"by": [], "ascending": []}
            
        # Select 1-2 columns for sorting
        all_cols = df.columns.tolist()
        num_sort_cols = min(random.randint(1, 2), len(all_cols))
        
        # Prefer numeric and datetime columns
        priority_cols = table_info["numeric_cols"] + table_info["datetime_cols"]
        if priority_cols:
            by_cols = random.sample(priority_cols, min(num_sort_cols, len(priority_cols)))
            if len(by_cols) < num_sort_cols:
                # If not enough, supplement from other columns
                other_cols = [c for c in all_cols if c not in priority_cols]
                by_cols.extend(random.sample(other_cols, min(num_sort_cols - len(by_cols), len(other_cols))))
        else:
            # Randomly select
            by_cols = random.sample(all_cols, num_sort_cols)
        
        # Randomly set ascending/descending
        ascending = [random.choice([True, False]) for _ in by_cols]
        
        return {"by": by_cols, "ascending": ascending}


class PivotOperation(Operation):
    """Pivot table operation"""
    
    compatible_dtypes = ["categorical", "numeric", "mixed"]
    
    def validate_params(self):
        """Validate parameter validity"""
        index = self.get_param("index")
        columns = self.get_param("columns")
        values = self.get_param("values")
        
        if not index or not columns or not values:
            raise ValueError("Pivot table operation must specify index, columns, and values parameters")
    
    def transform(self) -> pd.DataFrame:
        """Execute pivot table transformation"""
        df = self.table.copy()
        index = self.get_param("index")
        columns = self.get_param("columns")
        values = self.get_param("values")
        if isinstance(columns, list):
            columns = columns[0]
        if isinstance(index, list):
            index = index[0]
        if isinstance(values, list):
            values = values[0]
        aggfunc = self.get_param("aggfunc", "mean")
        
        try:
            pivot_table = pd.pivot_table(
                df, 
                index=index, 
                columns=columns, 
                values=values,
                aggfunc=aggfunc
            )
            
            # Reset index to make the result easier to handle
            pivot_table = pivot_table.reset_index()
            return pivot_table
        except Exception as e:
            raise ValueError(f"Pivot table transformation failed: {str(e)}")  # Debug information
            # print(f"Pivot table transformation failed: {str(e)}, returning original data")  # Debug information
            # return df

    def generate_params(self) -> Dict[str, Any]:
        df = self.table
        table_info = self.table_info
        """Generate pivot table parameters"""
        if (not table_info["categorical_cols"] or 
            not table_info["numeric_cols"] or 
            len(df.columns) < 3):
            # Pivot table requires at least one categorical column as index, one categorical column as columns, and one numeric column as values
            return {}
        
        # Select index column (prefer categorical columns)
        col_unique = []
        if table_info["categorical_cols"]:
            col_unique = [random.choice([col for col in table_info["categorical_cols"] if df[col].nunique() < 10])]
            index = [random.choice([col for col in table_info["categorical_cols"] if not col in col_unique])]
        else:
            index = [random.choice(df.columns.tolist())]
        
        # Select column name (cannot be the same as index)
        remaining_non_numeric = [c for c in table_info["categorical_cols"] if c not in index]
        remaining_non_numeric += [c for c in table_info["text_cols"] if c not in index]

        # Filter columns with less than 10 unique values
        remaining_non_numeric = [col for col in remaining_non_numeric if df[col].nunique() < 10]

        if remaining_non_numeric:
            # Prefer columns with fewer unique values
            columns = sorted(remaining_non_numeric, key=lambda col: df[col].nunique())[0:1]
        else:
            # If there are no additional non-numeric columns, use another column (not the same as index)
            other_cols = [c for c in df.columns if c not in index if df[c].nunique() < 10]
            if other_cols:
                columns = [random.choice(other_cols)]
            else:
                return {}  #

        # Ensure columns and values are not the same
        values_candidates = [c for c in table_info["numeric_cols"] if c not in columns]
        if not values_candidates:
            return {}  # If there are not enough numeric columns

        # Select value column (numeric column)
        values = [random.choice(values_candidates)]
        
        # Select aggregation function
        aggfunc = random.choice(["mean", "sum", "max", "min", "count"])
        
        return {
            "index": index[0],
            "columns": columns[0],
            "values": values[0],
            "aggfunc": aggfunc
        }



class StackOperation(Operation):
    """Stack operation"""
    
    compatible_dtypes = ["mixed"]
    
    def validate_params(self):
        """Validate parameter validity"""
        id_vars = self.get_param("id_vars", [])
        value_vars = self.get_param("value_vars", [])
        
        if not id_vars or not value_vars:
            raise ValueError("Stack operation must specify id_vars and value_vars parameters")
    
    def transform(self) -> pd.DataFrame:
        """Execute stack transformation"""
        df = self.table.copy()
        id_vars = self.get_param("id_vars", [])
        value_vars = self.get_param("value_vars", [])
        var_name = self.get_param("var_name", "variable")
        value_name = self.get_param("value_name", "value")
        
        try:
            # Use melt function for stacking
            result = pd.melt(
                df,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name
            )
            result[value_name] = result[value_name].astype("string")
            return result
        except Exception as e:
            raise ValueError(f"Stack transformation failed: {str(e)}")
            

    def generate_params(self) -> Dict[str, Any]:
        """Generate stack parameters"""
        df = self.table
        if len(df.columns) < 3:
            # Stacking requires at least one ID column and two value columns
            return {}
        
        # Select ID variables (1-2 columns)
        all_cols = df.columns.tolist()
        id_vars_count = min(random.randint(1, 2), len(all_cols) - 2)
        id_vars = random.sample(all_cols, id_vars_count)
        
        # Select value variables (at least 1 column, not the same as ID)
        remaining_cols = [c for c in all_cols if c not in id_vars]
        
        max_value_vars = min(len(remaining_cols), 5)
        value_vars_count = random.randint(1, max_value_vars)
        value_vars = random.sample(remaining_cols, value_vars_count)
        
        return {
            "id_vars": id_vars,
            "value_vars": value_vars
        }


class ExplodeOperation(Operation):
    """Explode operation"""
    
    compatible_dtypes = ["mixed"]
    
    def validate_params(self):
        """Validate parameter validity"""
        column = self.get_param("column")
        split_comma = self.get_param("split_comma", default=False)  # Add split_comma parameter
        if not column:
            raise ValueError("Explode operation must specify column parameter")
        df = self.table
        if column not in df.columns:
            raise ValueError(f"The specified column {column} does not exist")
        sample = df[column].dropna().iloc[0]
        
        if not isinstance(sample, (list, str)):
            raise ValueError(f"The column {column} needs to contain list or string data")
        
        # If it is a list type, the split_comma parameter is not needed
        if isinstance(sample, list):
            if split_comma:
                raise ValueError(f"The data in column {column} is already a list, no need to use the split_comma parameter")
        
        # If it is a string type and needs to check the split_comma parameter
        if isinstance(sample, str) and split_comma:
            if not all(isinstance(val, str) for val in df[column].dropna()):
                raise ValueError(f"The data in column {column} needs to be string")
    
    def transform(self) -> pd.DataFrame:
        """Execute explode transformation"""
        df = self.table.copy()
        column = self.get_param("column")
        split_comma = self.get_param("split_comma", default=False)  # Get parameter

        if column not in df.columns:
            return df
        
        try:
            sample = df[column].iloc[0]
            if isinstance(sample, list):
                return df.explode(column)
            elif isinstance(sample, str):
                if split_comma:
                    df[column] = df[column].apply(lambda x: x.split(',') if isinstance(x, str) else [x])
                else:
                    try:
                        df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                        return df.explode(column)
                    except (ValueError, SyntaxError):
                        df[column] = df[column].apply(lambda x: x.split() if isinstance(x, str) else [x])
                return df.explode(column)
            else:
                tmp_col = f"{column}_tmp"
                df[tmp_col] = df[column].apply(lambda x: [x, x] if pd.notna(x) else [])
                result = df.explode(tmp_col)
                result[column] = result[tmp_col]
                result = result.drop(columns=[tmp_col])
                return result
        except Exception as e:
            print(f"Error during explode: {e}")
            return df

    def generate_params(self) -> Dict[str, Any]:
        """Generate explode parameters"""
        df = self.table
        if not df.columns.tolist():
            return {}
        
        list_like_cols = []
        
        for col in df.columns:
            values = df[col].dropna().head(5).tolist()
            for val in values:
                if isinstance(val, (list, tuple)):
                    list_like_cols.append(col)
                    break
                if isinstance(val, str) and ',' in val:
                    list_like_cols.append(col)
                    break
        
        if not list_like_cols:
            column = random.choice(df.columns.tolist())
        else:
            column = random.choice(list_like_cols)
        
        # Determine if split_comma is needed if it is a list type
        sample = df[column].dropna().iloc[0]
        split_comma = isinstance(sample, str)  # If it is a string type, split_comma is True
        
        return {"column": column, "split_comma": split_comma}  # Return split_comma parameter
    

class WidetolongOperation(Operation):
    """Wide to long operation"""
    compatible_dtypes = ["mixed"]
    
    def validate_params(self):
        """Validate parameter validity"""
        subnames = self.get_param("subnames", [])
        i = self.get_param("i", [])
        sep = self.get_param("sep", "_")
        
        if not subnames or not i:
            raise ValueError("Must specify subnames and i parameters")
        if not isinstance(subnames, list) or not isinstance(i, list):
            raise TypeError("subnames and i must be lists")
        if sep and not isinstance(sep, str):
            raise TypeError("sep must be a string")

    def transform(self) -> pd.DataFrame:
        """Execute wide to long transformation"""
        df = self.table.copy()
        subnames = self.get_param("subnames", [])
        i = self.get_param("i", [])
        j = self.get_param("j", "variable")
        sep = self.get_param("sep", "_")
        suffix = self.get_param("suffix", r"\d+")
        
        try:
            # Use pandas' wide_to_long to achieve transformation [[7]]
            result = pd.wide_to_long(
                df,
                stubnames=subnames,
                i=i,
                j=j,
                sep=sep,
                suffix=suffix
            ).reset_index()
            for col in subnames:
                result[col] = result[col].astype("string")
            return result
        except Exception as e:
            raise ValueError(f"Wide to long transformation failed: {str(e)}")  # Debug information

    def generate_params(self) -> Dict[str, Any]:
        df = self.table
        cols = df.columns.tolist()
        valid_cols = [col for col in cols if any(sep in col for sep in [' ', '_', '-', '.', '/', ''])]
        
        # Count the shared prefixes and suffixes
        prefix_map = defaultdict(set)  # {prefix: {suffixes}}
        suffix_map = defaultdict(set)  # {suffix: {prefixes}}
        sep_candidates = defaultdict(int)
        
        for col in valid_cols:
            possible_splits = []
            # Prefer splitting from the right side (ensure the suffix is the last part)
            for sep in [' ', '_', '-', '.', '/']:
                if sep in col:
                    # Try splitting once from the right side
                    parts = col.rsplit(sep, 1)
                    if len(parts) == 2:
                        possible_splits.append((sep, parts[0], parts[1]))
            
            # Implicit split handling (letter-number boundary)
            if not possible_splits:
                match = re.match(r'^([a-zA-Z\s]+)(\d+)$', col)  # Allow spaces in the prefix
                if match:
                    possible_splits.append(('', match.group(1).strip(), match.group(2)))
            
            # Record split results
            for sep, first, second in possible_splits:
                prefix_map[(first, sep)].add(second)
                suffix_map[(second, sep)].add(first)
                sep_candidates[sep] += 1
        
        # Evaluate candidate quality
        def evaluate_candidates(candidates):
            return {
                key: len(values) 
                for key, values in candidates.items()
                if len(values) >= 2  # Shared by at least two different parts
            }
        
        prefix_scores = evaluate_candidates(prefix_map)
        suffix_scores = evaluate_candidates(suffix_map)
        
        # Intelligently select the best direction (prefer suffix, but handle prefix structure)
        subname_type = 'suffix'
        best_keys = [] 
        if  max(prefix_scores.values(), default=0) >= max(suffix_scores.values(), default=0):
            subname_type = 'prefix'
            best_keys = [
                key for key in prefix_scores.keys()
                if prefix_scores[key] == max(prefix_scores.values())
            ]
        else:
            if suffix_scores:
                best_keys = [
                    key for key in suffix_scores.keys()
                    if suffix_scores[key] == max(suffix_scores.values())
                ]
        if not best_keys:
            return {"subnames": [], "i": cols, "j": "var", "sep": '', "suffix": ''}
        best_sep = ' '
        if sep_candidates:
            best_sep = max(sep_candidates.items(), key=lambda x: (x[1], x[0] != ''))[0]
        
        # Extract subnames and corresponding separators
        subnames = []
        for key in best_keys:
            subname, sep = key
            subnames.append(subname.strip())  # Remove possible spaces
            best_sep = sep  # Prefer high-frequency separators
        
        # Remove duplicates and filter out empty values
        subnames = list(set([s for s in subnames if s]))
        
        # Determine id_vars (non-subname columns)
        id_vars = []
        for col in cols:
            is_subname_col = False
            for s in subnames:
                # Prefix type: column name starts with "subname<sep>"
                if subname_type == 'prefix':
                    if best_sep:
                        if col.startswith(f"{s}{best_sep}"):
                            is_subname_col = True
                            break
                    else:
                        # Implicit split (letter-number boundary)
                        if col.startswith(s) and len(col) > len(s) and col[len(s):].isdigit():
                            is_subname_col = True
                            break
                # Suffix type: column name ends with "<sep>subname"
                else:
                    if best_sep:
                        if col.endswith(f"{best_sep}{s}"):
                            is_subname_col = True
                            break
                    else:
                        # Implicit split (letter-number boundary)
                        if col.endswith(s) and len(col) > len(s) and col[:-len(s)].isalpha():
                            is_subname_col = True
                            break
            if not is_subname_col:
                id_vars.append(col)
        
        return {
            "subnames": subnames,
            "i": id_vars,
            "j": "var",
            "sep": best_sep,
            "suffix": r"\w+"  # Default to alphanumeric matching
        }

class JoinOperation(Operation):
    """Improved join operation, handling single table splitting in parameter generation phase"""
    compatible_dtypes = ["mixed"]
        
    def validate_params(self):
        """Validate parameter validity"""
        required = ['left_table', 'right_table', 'left_on', 'right_on']
        missing_params = []
        for p in required:
            param = self.get_param(p)
            if param is None or (isinstance(param, pd.DataFrame) and param.empty):
                missing_params.append(p)
        if missing_params:
            raise ValueError(f"Missing required parameters or parameters are empty: {missing_params}")

    def transform(self) -> pd.DataFrame:
        
        """Execute join transformation (only handle multiple tables)"""
        if not isinstance(self.table, List):
            raise ValueError("Two tables are required for joining")
        left_table = self.table[0].copy()
        right_table = self.table[1].copy()
        # on_col = self.get_param("on")
        # if isinstance(on_col, List):
        #     on_col = on_col[0]
        left_on = self.get_param("left_on")
        right_on = self.get_param("right_on")
        how = self.get_param("how", "inner")
        suffixes = self.get_param("suffixes", ('_x', '_y'))
        l_dtype = left_table[left_on].dtype
        r_dtype = right_table[right_on].dtype
        # If the dtypes on both sides are different, force them to be converted to str
        if l_dtype != r_dtype:
            left_table[left_on]  = left_table[left_on].astype(str)
            right_table[right_on] = right_table[right_on].astype(str)
        return left_table.merge(right_table, left_on=left_on, right_on=right_on, how=how, suffixes=suffixes)
        # if isinstance(on_col, str):
        #     return left_table.merge(right_table, on=on_col, how=how, suffixes=suffixes)
        # elif isinstance(on_col, dict):
        #     for key, col in on_col.items():
        #         if key == 0:
        #             left_on = col
        #         elif key == 1:
        #             right_on = col
        #     l_dtype = left_table[left_on].dtype
        #     r_dtype = right_table[right_on].dtype
        #     # If the dtypes on both sides are different, force them to be converted to str
        #     if l_dtype != r_dtype:
        #         left_table[left_on]  = left_table[left_on].astype(str)
        #         right_table[right_on] = right_table[right_on].astype(str)
        #     return left_table.merge(right_table, left_on=left_on, right_on=right_on, how=how, suffixes=suffixes)

    def generate_params(self, join_cols) -> Dict[str, Any]:
        """Intelligently generate join parameters (handle single table splitting)"""
        tables = self.table
        if isinstance(tables, pd.DataFrame):
                split_col = self._select_split_column(tables)
                
                # 2. Split other columns into two groups
                other_cols = [col for col in tables.columns if col != split_col]
                random.shuffle(other_cols)  # Randomly shuffle column order
                
                # Allocate columns to left and right tables by proportion
                split_point = len(other_cols) // 2
                left_cols = other_cols[:split_point] + [split_col]
                right_cols = other_cols[split_point:] + [split_col]
                
                # Get unique values of the join key column
                unique_keys = tables[split_col].unique()
                
                # Randomly select some key values for the left table and others for the right table
                np.random.shuffle(unique_keys)
                split_key_point = len(unique_keys) // 2
                left_keys = set(unique_keys[:split_key_point])
                right_keys = set(unique_keys[split_key_point:])
                
                overlapping_keys = set(np.random.choice(unique_keys, size=min(len(unique_keys) // 3, len(unique_keys)), replace=False))
                left_keys.update(overlapping_keys)
                right_keys.update(overlapping_keys)
                
                # Create masks
                left_mask = tables[split_col].isin(left_keys)
                right_mask = tables[split_col].isin(right_keys)
                
                # Ensure both left and right tables have data
                if left_mask.sum() == 0:
                    left_mask = tables[split_col] == unique_keys[0]
                if right_mask.sum() == 0:
                    right_mask = tables[split_col] == unique_keys[-1]
                
                # Split data
                left = tables[left_cols].copy()
                right = tables[right_cols].copy()
                
                # Apply masks
                left = left[left_mask].reset_index(drop=True)
                right = right[right_mask].reset_index(drop=True)
                
                self.table = [left, right]
                return {
                    'left_table': left.columns.to_list(),
                    'right_table': right.columns.to_list(),
                    'left_on': split_col,
                    'right_on': split_col,
                    'how': random.choice(["inner", "left", "right", "outer"]),
                    'suffixes': random.choice([('_x', '_y'),('_left', '_right')])
                }
        else:
            # Multi-table join logic
            left = tables[0]
            right = tables[1]
            if join_cols:
                # If join columns are specified, use the specified columns
                for key, value in join_cols.items():
                    if key == 0:
                        left_on = value
                    elif key == 1:  
                        right_on = value
            else:
                common_cols = list(set(left.columns) & set(right.columns))    
                left_on = random.choice(common_cols)
                right_on = left_on
            return {
                'left_table': left.columns.to_list(),
                'right_table': right.columns.to_list(),
                'left_on': left_on,
                'right_on': right_on,
                'how': random.choice(["inner", "left", "right", "outer"]),
                'suffixes': random.choice([('_x', '_y'),('_left', '_right')])
            }

    def _select_split_column(self, df: pd.DataFrame) -> str:
        """Intelligently select split column"""
        candidates = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if 0.2 < unique_ratio < 0.8:
                candidates.append(col)
        return random.choice(candidates) if candidates else df.columns[0]


class UnionOperation(Operation):
    """Union operation supporting single table splitting, capable of handling intelligent merging with inconsistent column order"""
    compatible_dtypes = ["mixed"]
    
    def validate_params(self):
        """Validate required parameters"""
        required = ['how']
        missing_params = [p for p in required if not self.get_param(p)]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

    def transform(self) -> pd.DataFrame:
        """Execute union operation"""
        if not isinstance(self.table, list) or len(self.table) != 2:
            raise ValueError("Two tables are required for union operation")
        
        left, right = self.table.copy()
        how = self.get_param("how", "all")
        
        # Intelligent column alignment (when column order is inconsistent)
        try:
            combined = pd.concat([left, right], axis=0, ignore_index=True)
        except ValueError:
            # Handle column type mismatch
            right = right.astype(left.dtypes.to_dict())
            combined = pd.concat([left, right], axis=0, ignore_index=True)
        
        return combined.drop_duplicates() if how == "distinct" else combined

    def generate_params(self) -> Dict[str, Any]:
        """Generate union parameters (support single table splitting)"""
        tables = self.table
        
        if isinstance(tables, pd.DataFrame):
            # Single table splitting logic
            df = tables
            if len(df) < 2:
                raise ValueError("At least two rows of data are required for splitting")
            
            # Generate two subtables with overlapping data
            base_mask = np.random.rand(len(df)) < 0.6
            overlap_mask = np.random.rand(len(df)) < 0.3
            
            left = df[base_mask | overlap_mask].copy()
            right = df[(~base_mask) | overlap_mask].copy()
            
            # Randomly shuffle the column order of the right table
            if random.choice([True, False]):
                right = right.sample(frac=1, axis=1)
            
            # Add random null values to test robustness
            for col in random.sample(list(right.columns), k=random.randint(0, len(right.columns))):
                right.loc[random.sample(list(right.index), k=int(len(right)*0.1)), col] = None
            
            self.table = [left.reset_index(drop=True), right.reset_index(drop=True)]
            
            return {
                'left_table': left.columns.tolist(),
                'right_table': right.columns.tolist(),
                'how': random.choice(["all", "distinct"])
            }
        else:
            # Multi-table processing logic
            left, right = tables
            # Align column data types
            for col in right.columns:
                if col in left.columns and right[col].dtype != left[col].dtype:
                    right[col] = right[col].astype(left[col].dtype)
            
            return {
                'left_table': left.columns.tolist(),
                'right_table': right.columns.tolist(),
                'how': random.choice(["all", "distinct"])
            }

    def _adjust_columns(self, df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        """Align column order"""
        return df[reference.columns] if all(col in df.columns for col in reference.columns) else df
     
class TransposeOperation(Operation):
    """Transpose operation"""
    
    compatible_dtypes = ["numeric", "categorical", "datetime", "mixed", "text"]
    
    def validate_params(self):
        """Validate parameter validity"""
        # Transpose operation does not accept any parameters
        if self.params:
            raise ValueError(f"Transpose operation does not accept parameters: {list(self.params.keys())}")
    
    def transform(self) -> pd.DataFrame:
        """Execute transpose transformation"""
        try:
            # Directly return the transposed DataFrame
            df_t = self.table.transpose()
            new_columns = df_t.iloc[0].tolist()
            df_t = df_t.iloc[1:]
            df_t.insert(0, self.table.columns[0], df_t.index)
            df_t.columns = [self.table.columns[0]] + new_columns  # Set new column names
            df_t = df_t.reset_index(drop=True)  # Reset row index
            return df_t
        except Exception:
            # If transpose fails, return the original table
            return self.table

    def generate_params(self) -> Dict[str, Any]:
        """Generate transpose parameters"""
        # Transpose operation does not require parameters, always return an empty dictionary
        return {}
    
class RenameOperation(Operation):
    """Rename column names using LLM-based synonym replacement"""

    compatible_dtypes = ["numeric", "categorical", "datetime", "text", "mixed"]

    def validate_params(self):
        """Validate parameters"""
        rename_map = self.get_param("rename_map", {})
        if not isinstance(rename_map, dict):
            raise ValueError("rename_map parameter must be a dictionary")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in rename_map.items()):
            raise ValueError("rename_map keys and values must be strings")

    def transform(self) -> pd.DataFrame:
        """Apply the rename transformation"""
        df = self.table.copy()
        rename_map = self.get_param("rename_map", {})
        return df.rename(columns=rename_map)

    def generate_params(self) -> Dict[str, Any]:
        """Generate rename_map using LLM synonyms in JSON format"""
        df = self.table
        columns = df.columns.tolist()

        if not columns:
            return {"rename_map": {}}

        num_cols_to_rename = random.randint(1, min(3, len(columns)))
        cols_to_rename = random.sample(columns, num_cols_to_rename)

        # JSON-based prompt
        prompt = (
            "You are given a list of column names. For each column name, generate a meaningful and readable synonym.\n"
            "Return the result as a JSON object where the keys are the original column names and the values are the new names.\n\n"
            f"Column names: {cols_to_rename}\n\n"
            "Example output:\n"
            "{\n"
            "  \"gender\": \"sex\",\n"
            "  \"income\": \"earnings\"\n"
            "}"
        )

        # Setup LLM client
        from src.llm.client import LLMClient
        from src.config import LLM_CONFIG
        from src.baseline.llm_prompt.parsers.content_parser import ContentParser
        cfg = LLM_CONFIG.copy()
        m = cfg.get("model_name", "default_model")
        llm = LLMClient(config=cfg)
        response = llm.generate(prompt)

        # Try to parse JSON safely
        try:
            rename_map = ContentParser.parse_content(str(response).strip())
            # Only keep mappings for selected columns
            rename_map = {k: v for k, v in rename_map.items() if k in cols_to_rename}
        except Exception:
            # print(f"Failed to parse JSON response: {response}")
            rename_map = {}

        return {"rename_map": rename_map}

class DropNullsOperation(Operation):
    """Drop rows with missing values operation, reset index after execution"""

    compatible_dtypes = ["numeric", "categorical", "datetime", "mixed", "text"]

    def validate_params(self):
        """Validate parameter validity"""
        subset = self.get_param("subset", None)
        how = self.get_param("how", 'any')
        df = self.table
        if how not in ['any', 'all']:
            raise ValueError("The 'how' parameter must be 'any' or 'all'")
        if subset is not None:
            missing = [c for c in subset if c not in df.columns]
            if missing:
                raise ValueError(f"The following columns do not exist in the table and cannot be used for dropna: {missing}")

    def transform(self) -> pd.DataFrame:
        """Execute drop missing values transformation, and reset index"""
        df = self.table.copy()
        subset = self.get_param("subset", None)
        how = self.get_param("how", 'any')
        try:
            cleaned = df.dropna(subset=subset, how=how)
            return cleaned.reset_index(drop=True)
        except Exception as e:
            # Return the original table (reset original index) in case of error
            # return df.reset_index(drop=True)
            raise ValueError(f"Drop missing values transformation failed: {str(e)}")  # Debug information

    def generate_params(self) -> Dict[str, Any]:
        """Generate drop missing values parameters"""
        df = self.table
        cols = df.columns.tolist()
        if not cols:
            return {"subset": None, "how": 'any'}

        subset = None
        if random.random() < 0.5:
            num = min(random.randint(1, 2), len(cols))
            subset = random.sample(cols, num)

        how = random.choice(['any', 'all'])
        return {"subset": subset, "how": how}


class DeduplicateOperation(Operation):
    """Deduplication operation, reset index after execution"""

    compatible_dtypes = ["numeric", "categorical", "datetime", "mixed", "text"]

    def validate_params(self):
        """Validate parameter validity"""
        subset = self.get_param("subset", None)
        keep = self.get_param("keep", 'first')
        df = self.table
        if keep not in ['first', 'last']:
            raise ValueError("The 'keep' parameter must be 'first' or 'last'")
        if subset is not None:
            missing = [c for c in subset if c not in df.columns]
            if missing:
                raise ValueError(f"The following columns do not exist in the table and cannot be used for drop_duplicates: {missing}")

    def transform(self) -> pd.DataFrame:
        """Execute deduplication transformation, and reset index"""
        df = self.table.copy()
        subset = self.get_param("subset", None)
        keep = self.get_param("keep", 'first')
        try:
            deduped = df.drop_duplicates(subset=subset, keep=keep)
            return deduped.reset_index(drop=True)
        except Exception as e:
            # Return the original table (reset original index) in case of error
            # return df.reset_index(drop=True)
            raise ValueError(f"Deduplication transformation failed: {str(e)}")  # Debug information

    def generate_params(self) -> Dict[str, Any]:
        """Generate deduplication parameters"""
        df = self.table
        cols = df.columns.tolist()
        if not cols:
            return {"subset": None, "keep": 'first'}

        subset = None
        if random.random() < 0.5:
            num = min(random.randint(1, 2), len(cols))
            subset = random.sample(cols, num)

        keep = random.choice(['first', 'last'])
        return {"subset": subset, "keep": keep}

class TopKOperation(Operation):
    """Top-K operation: directly take the top k rows without sorting by columns"""
    compatible_dtypes = ["numeric", "categorical", "datetime", "mixed"]

    def validate_params(self):
        k = self.get_param("k", None)
        if k is None or not isinstance(k, int) or k <= 0:
            raise ValueError("Must specify a positive integer k")

    def transform(self) -> pd.DataFrame:
        df = self.table.copy()
        k = self.get_param("k", 0)
        try:
            return df.head(k)
        except Exception as e:
            raise ValueError(f"Top-K transformation failed: {str(e)}")  # Debug information

    def generate_params(self) -> Dict[str, Any]:
        df = self.table
        n_rows = len(df)
        if n_rows == 0:
            return {"k": 0}
        k = random.randint(1, min(10, n_rows))
        return {"k": k}


class SelectColOperation(Operation):
    """Select columns operation"""
    compatible_dtypes = ["numeric", "categorical", "datetime", "mixed"]

    def validate_params(self):
        cols = self.get_param("columns", [])
        if not cols:
            raise ValueError("Must specify columns to select (columns)")

    def transform(self) -> pd.DataFrame:
        df = self.table.copy()
        cols = self.get_param("columns", [])
        valid = [c for c in cols if c in df.columns]
        return df[valid]

    def generate_params(self) -> Dict[str, Any]:
        df = self.table
        all_cols = df.columns.tolist()
        if not all_cols:
            return {"columns": []}
        n = random.randint(1, len(all_cols))
        cols = random.sample(all_cols, n)
        return {"columns": cols}


class CastTypeOperation(Operation):
    """Type conversion operation: supports string↔numeric, datetime↔string, integer↔float, boolean↔others"""
    compatible_dtypes = ["numeric", "categorical", "datetime", "mixed"]

    def validate_params(self):
        col = self.get_param("column", None)
        dtype = self.get_param("dtype", None)
        if col is None:
            raise ValueError("Must specify the column to convert type (column)")
        if dtype not in ["int", "float", "str", "datetime64", "bool"]:
            raise ValueError(f"Unsupported target type (dtype): {dtype}")

    def transform(self) -> pd.DataFrame:
        df = self.table.copy()
        col = self.get_param("column")
        dtype = self.get_param("dtype")
        series = df[col]
        try:
            if dtype == "datetime64":
                # Convert to datetime, unify type
                converted = pd.to_datetime(series, errors="coerce")
                df[col] = converted
            elif dtype == "bool":
                if pd.api.types.is_numeric_dtype(series):
                    df[col] = series != 0
                elif pd.api.types.is_object_dtype(series):
                    mapped = series.astype(str).str.lower().map({'true': True, 'false': False})
                    df[col] = mapped.fillna(False)
                else:
                    df[col] = series.astype(bool, errors="ignore")
            elif dtype == "int":
                # Force conversion to integer, non-integer or missing values are filled with 0
                numeric = pd.to_numeric(series, errors="coerce").fillna(0)
                df[col] = numeric.astype(int)
            elif dtype == "float":
                numeric = pd.to_numeric(series, errors="coerce")
                df[col] = numeric.astype(float)
            else:  # str
                df[col] = series.astype(str)
        except Exception:
            raise ValueError(f"Type conversion failed: {col} -> {dtype}")
        return df
    
    def generate_params(self) -> Dict[str, Any]:
        table = self.table
        dtypes = table.dtypes
        columns = table.columns.tolist()

        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        text_cols = []

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
                    continue  

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

        # String -> Numeric (check if it can be converted)
        for col in text_cols:
            if table[col].apply(is_number_like).mean() > 0.8:
                conversions.append((col, 'float'))

        # Datetime -> String
        for col in datetime_cols:
            conversions.append((col, 'str'))

        # String -> Datetime (simple sample check)
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
            return {"column": None, "dtype": None}

        col, target_type = random.choice(conversions)
        return {"column": col, "dtype": target_type}


# Register all operations
OperationFactory.register("filter", FilterOperation)
OperationFactory.register("groupby", GroupByOperation)
OperationFactory.register("sort", SortOperation)
OperationFactory.register("pivot", PivotOperation)
OperationFactory.register("unpivot", StackOperation)
OperationFactory.register("explode", ExplodeOperation)
OperationFactory.register("wide_to_long", WidetolongOperation)
OperationFactory.register("union", UnionOperation)
OperationFactory.register("join", JoinOperation)
OperationFactory.register("transpose", TransposeOperation)
OperationFactory.register("dropna", DropNullsOperation)
OperationFactory.register("deduplicate", DeduplicateOperation)
OperationFactory.register("topk", TopKOperation)
OperationFactory.register("select", SelectColOperation)
OperationFactory.register("cast", CastTypeOperation)
OperationFactory.register("rename", RenameOperation)