"""
Intent generation agent, generates natural language instructions based on tables and transformation chains, and verifies that the intent covers all transformation operations.
"""
from typing import List
from src.agents.base import Agent
import json

class IntentAgent(Agent):
    """
    Intent generation agent, generates natural language instructions based on tables and transformation chains, and verifies whether the generated intent covers all operations in the transformation chain.
    If the verification fails, the LLM will rewrite the intent.
    """
    
    def __init__(self, config, llm_client):
        """
        Initialize the intent generation agent.
        Args:
            config: Agent configuration
            llm_client: LLM client
        """
        super().__init__(config)
        self.llm_client = llm_client
    
    def run(self, T_input, T_target, transform_chain):
        """
        Generate natural language instructions and intents based on input tables, target tables, and transformation chains, and verify the completeness of the intent.
        Args:
            T_input: Input tables, can be multiple
            T_target: Target table
            transform_chain: Transformation chain
            language: Language
        Returns:
            tuple(str instruction, str intent)
        """
        # Generate task description instruction
        instruction_prompt = self._prepare_prompt_instruction(T_input, T_target, transform_chain)
        response = self.llm_client.generate(instruction_prompt)
        instruction = self._extract_instruction(response)

        # Generate initial intent
        intent_prompt = self._prepare_prompt_intent(instruction, transform_chain)
        initial_intent = self.llm_client.generate(intent_prompt).strip()
        initial_intent = initial_intent.replace("意图表述：", "").replace("User Intent:", "").strip()

        # Verify whether the intent covers all transformation operations
        validation_prompt = self._prepare_prompt_validation(transform_chain, initial_intent, instruction)
        validation_resp = self.llm_client.generate(validation_prompt)

        # Parse validation results
        try:
            result = json.loads(validation_resp)
            is_valid = str(result.get("is_valid", "false")).lower() == "true"
            rewritten_intent = result.get("intent", initial_intent)
        except Exception:
            # If parsing fails, consider it invalid
            is_valid = True
            rewritten_intent = initial_intent

        return instruction, initial_intent, is_valid, rewritten_intent
    
    def _prepare_prompt_instruction(self, T_input, T_target, transform_chain):
        """
        Prepare LLM prompt to generate operation instructions.
        """
        if not isinstance(T_input, List):
            T_input = [T_input]
        input_table_str = []
        for idx, table in enumerate(T_input, 1):
            header = f"table_{idx}：\n"
            content = table.head(5).to_string()
            input_table_str.append(header + content)
        target_table_str = T_target.head(5).to_string()
        transform_chain_str = self._format_transform_chain(transform_chain)
        prompt = f"""
You are a data transformation expert. I have some related input tables and a target table, where the target table is obtained by transforming the input tables. Based on the transformation relationship between them, please generate a clear natural language instruction that describes how to transform the input tables into the target table.

The transformation operations and their detailed parameters are as follows:  
{transform_chain_str}

Input Tables (First 10 Rows):  
{input_table_str}

Target Table (First 10 Rows):  
{target_table_str}

Please generate a clear and natural data transformation instruction in English. The instruction should explicitly describe the required transformation steps and clearly state the table names involved, without mentioning specific programming languages or function names. Use terminology from the data analysis domain and consider the purpose and effect of the operations.
Your instructions just need to clearly describe the conversion chain without describing additional operations.
Your instruction should follow the format:  
Instruction: [Your data transformation instruction]
        """
        return prompt
    
    def _prepare_prompt_intent(self, task_instruction, transform_chain):
        """
        Prepare LLM prompt to generate user intent.
        """
        transform_chain = self._format_transform_chain(transform_chain)
        GENERALIZED_INTENT_TEMPLATE = f"""
Based on the following data transformation task description, generate a natural language statement expressing the user's intent.
Concise, Action-Oriented Language: Focus on the core actions and remove unnecessary details. Keep the language clear and direct to highlight the transformation intent.
Clarification of Key Tables and Columns: Maintain essential table names and columns, but express them in a natural, straightforward way.
Simplified Descriptions of Complex Steps: Emphasize the main objectives (sorting, filtering, deduplication) without diving into excessive details, unless they are crucial for the context.
Necessary details need to be preserved such as the suffix of the join, the way the de-duplication operation is performed (first or last), etc.

Here are some examples:
---
Task Description: To transform the input tables into the target table, follow these steps: 1. Begin by performing an inner join between table_1 and table_2 using the allergy name column from table_1 and the allergy column from table_2. This will combine records from both tables where there is a match on these columns, while including the allergy name and allergy type from table_1 along with the stuid from table_2. 2. Next, group the resulting dataset by the allergy name (now included in the joined table) and aggregate the data by counting the number of unique stuid entries for each allergy name. This will give you the total number of students associated with each allergy. 3. After aggregating, sort the grouped data first by the count of stuid in ascending order and then by allergy name in descending order. This will organize the data based on the number of students and the names of the allergens. 4. From the sorted data, select the top 7 entries based on the highest counts of students. This step ensures that we focus only on the most significant allergens. 5. Rename the columns in the resulting dataset by changing allergy name to allergen and stuid to student ID to make the column names more intuitive. 6. Apply a filter to retain only those records where the student ID (which now represents the count of students) is greater than or equal to 3. This will help in identifying the allergens that have a notable number of students associated with them. 7. Remove any duplicate entries from the filtered dataset to ensure that each allergen-student ID combination is unique. 8. Finally, perform a sort on the deduplicated data by student ID in ascending order and allergen in descending order to achieve the desired final format. Following these steps will yield a table that lists allergens along with the count of students associated with each, structured as specified in the target table.
User Intent: Start by performing an inner join between table_1 and table_2 on 'allergy name' and 'allergy', with suffixes '_left' and '_right'. Then, group the data by allergy name and count the number of 'stuid' entries for each allergen to determine the number of students associated with each allergy. After grouping, sort the data first by the student count in ascending order and then by allergy name in descending order. Select the top 7 entries. Rename the columns to change allergy name to allergen and stuid to student ID for clarity. Apply a filter to keep only the records where the student ID is 3 or greater. Deduplicate the data, keeping the first occurrence of each duplicate entry to ensure uniqueness. Finally, sort the deduplicated dataset by student ID in ascending order and allergen in descending order to produce the final result.
---
Task Description: First, combine the two input tables, table_1 and table_2, by performing a union operation to consolidate all records, including duplicates. Next, pivot the resulting table to reorganize the data, setting the station names (STN_NAM) as the index, the data provider (DATA_PVDR) as the columns, and using the minimum longitude (LONGITUDE) as the values. After pivoting, rename the column STN_NAM to Station Name. Then, filter the table to keep only the rows where the data provider is "NAV CANADA". Following this, remove any rows that contain missing values in the "NAV CANADA" column. Convert the data type of the "NAV CANADA" column to string. Next, ensure there are no rows where "NAV CANADA" is equal to itself (this condition might be meant for data cleansing or error checking). Finally, deduplicate the entries based on the "NAV CANADA" column while keeping the last occurrence of each duplicate. The result will be your target table with the columns DATA_PVDR and NAV CANADA.
User Intent: Begin by performing a union operation on table_1 and table_2 to consolidate all records, including duplicates. Then, pivot the resulting table with the station names (STN_NAM) as the index, the data provider (DATA_PVDR) as the columns, and use the minimum longitude (LONGITUDE) as the values. Rename the STN_NAM column to "Station Name" for clarity. Next, select the only column "NAV CANADA", and remove any rows with missing values in the "NAV CANADA" column. Convert the "NAV CANADA" column to a string data type and ensure that there are no rows where "NAV CANADA" is equal to itself. Finally, deduplicate the data based on the "NAV CANADA" column, keeping the last occurrence of each duplicate entry.
---
Task Description: First, reshape the data from the wide format to a long format by selecting the columns related to 'PUZZLE B' and 'PUZZLE A', while keeping the specified index columns intact. After transforming the data to a long format, you can apply the explode operation. This operation will split any column containing comma-separated values into individual rows. Next,transforms data from wide format to long format,it keeps the columns in id_vars unchanged and stacks the values from value_vars ("PUZZLE A" and "PUZZLE B") into two new columns: one for the variable names and another for the values.
User Intent: First,reshape the data by collapsing columns that start with "PUZZLE B" or "PUZZLE A" into a long format, while keeping the specified index columns ("Index", "Where are we?") unchanged. The original suffixes from the column names are extracted into a new column called var, using a space as the separator and matching suffixes with a word character pattern (\w+).Then, Explode the "PUZZLE B" column to create separate rows for each puzzle listed, ensuring that each puzzle is split by commas first. Next,transforms data from wide format to long format,the columns specified in id_vars ("Index", "Where are we?") remain unchanged and serve as identifiers for each row. The values in value_vars ("PUZZLE A" and "PUZZLE B") are then stacked into two new columns: one for the variable names and another for the values.
---

Now, based on the following task description, generate a user intent statement:
Transformation Chain: {transform_chain}
Task Description: {task_instruction}

Please output only the intent statement, without explanation or numbering.
"""
        return GENERALIZED_INTENT_TEMPLATE
    
    def _prepare_prompt_validation(self, transform_chain, intent_text, instruction):
        """
        Prepare LLM prompt to verify whether the intent includes all transformation chain operations and rewrite if necessary.
        """
        transform_chain_str = self._format_transform_chain(transform_chain)

        prompt = f"""
Task Background: The user has generated an initial natural language description from a transformation chain, and then used an LLM to generate a user intent statement based on that initial description.
1. **Transformation Chain**: {transform_chain_str}
2. **Initial Natural Language Description**: {instruction}
3. **Generated Intent**: {intent_text}

Task Requirement: Assume you are a data preparation expert. Based on the current intent, can you infer the correct conversion chain, including the details of the parameters?

Output Requirements:
- If the intent allows you to infer a complete and reasonable transformation chain, output:
{{
"is_valid": "true",
"intent": "{intent_text}"
}}
- Otherwise, output:
{{
"is_valid": "false",
"intent": "[Rewritten Intent]"
}}

Please return the result in strict JSON format with no additional explanations.
"""

        return prompt

    
    def _format_transform_chain(self, transform_chain):
        """
        Format the transformation chain into a readable string.
        """
        if not transform_chain:
            return "No transformation operations"
        if isinstance(transform_chain, str):
            return f"- {transform_chain}"
        elif isinstance(transform_chain, list):
            result = []
            for i, transform in enumerate(transform_chain):
                if isinstance(transform, str):
                    result.append(f"{i+1}. {transform}")
                elif isinstance(transform, dict):
                    op = transform.get("op", "Unknown operation")
                    params = transform.get("params", {})
                    op_tables = transform.get("op_tables", [])
                    param_strs = []
                    if op == "join" or op == "union":
                        for key, value in params.items():
                            if key == "left_table" and op_tables:
                                param_strs.append(f"{key}: {op_tables[0]}")
                            elif key == "right_table" and len(op_tables) > 1:
                                param_strs.append(f"{key}: {op_tables[1]}")
                            else:
                                param_strs.append(f"{key}: {value}")
                    else:        
                        for key, value in params.items():
                            if isinstance(value, dict):
                                nested_strs = [f"'{k}': '{v}'" for k, v in value.items()]
                                param_strs.append(f"{key}: {{{', '.join(nested_strs)}}}")
                            else:
                                param_strs.append(f"{key}: {value}")
                    param_str = ", ".join(param_strs) if param_strs else "No parameters"
                    table_str = ", ".join(op_tables) if op_tables else "No tables"
                    result.append(f"{i+1}. {op} ({param_str}) (op_tables: {table_str})")
            return "\n".join(result)
        return str(transform_chain)
    def _extract_instruction(self, response):
        """
        Extract instruction from LLM response.
        
        Args:
            response: LLM response
            
        Returns:
            str: Extracted instruction
        """
        # Try to find the content after "Instruction:"
        if "指令:" in response:
            instruction = response.split("指令:")[1].strip()
        elif "Instruction:" in response:
            instruction = response.split("Instruction:")[1].strip()
        else:
            # If no marker is found, use the entire response
            instruction = response.strip()
        
        return instruction