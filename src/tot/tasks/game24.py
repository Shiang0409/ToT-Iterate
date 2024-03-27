import re
import os
import sympy
import pandas as pd
from src.tot.tasks.base import Task, DATA_PATH
from src.tot.prompts.game24 import * 


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1] # 去除空格，按行分割，取最後一行
    return last_line.split('left: ')[-1].split(')')[0] # 提取剩餘的數字字符串


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    """
    Game24任務類，繼承自Task基類

    Attributes:
        data (list): 存儲24遊戲問題的列表
        value_cache (dict): 用於緩存值的字典
        steps (int): 24遊戲的步數
        stops (list): 停止標誌列表
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__() # 呼叫父類別 Task
        path = os.path.join(DATA_PATH, '24', file) # 構建文件路徑
        self.data = list(pd.read_csv(path)['Puzzles']) # 讀取並存儲問題數據
        self.value_cache = {} # 初始化值緩存字典
        self.steps = 4 # 24遊戲的步數
        self.stops = ['\n'] * 4 # 停止標誌列表

    def __len__(self) -> int:
        return len(self.data) #返回問題數量
    
    def get_input(self, idx: int) -> str:
        return self.data[idx] # 獲取指定索引處的輸入數據

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0] # 提取表達式
        numbers = re.findall(r'\d+', expression) # 使用正則表達式查找數字
        problem_numbers = re.findall(r'\d+', self.data[idx]) # 查找問題數字
        if sorted(numbers) != sorted(problem_numbers): # 如果表達式中的數字不等於問題中的數字
            return {'r': 0} # 返回錯誤
        try:
            print(f'r: {sympy.simplify(expression)}\n')
            return {'r': int(sympy.simplify(expression) == 24)} # 測試表達式是否等於24
        except Exception as e:
            print(f'e: {e}\n')
            return {'r': 0} # 出現異常返回錯誤
            
    @staticmethod # 不需要多設變數去實例化即可調用
    def standard_prompt_wrap(x: str, y:str='') -> str:
        print(f'standard prompt: {standard_prompt.format(input=x) + y}\n')
        return standard_prompt.format(input=x) + y # 根據輸入 x 和 y 構造標準提示訊息

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y # 根據輸入 x 和 y 構造 cot 提示訊息
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x) # 如果 y 為空，則使用 x 作為當前數字；否則使用 y
        if current_numbers == '24': # 如果當前數字為 '24'，則構造 cot 提示訊息；否則構造提議提示訊息
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            print(f'cot prompt response:{[prompt]}\n')
        else:
            prompt = propose_prompt.format(input=current_numbers)
            print(f'propose prompt response:{[prompt]}\n')
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1] # 獲取 y 中最後一行
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '') # 如果最後一行不包含 'left: '，則為最後一步，構造值提示訊息，並返回
            print(f'value_last_step_prompt: {[value_last_step_prompt.format(input=x, answer=ans)]}\n')
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y) # 否則從 y 中提取當前剩餘的數字，並構造值提示訊息
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float: # value 多個數據縮成一個值
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower(): # 如果 y 的最後一行包含 4 個 '\n' 且不包含 'answer'，則返回 0
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs] # 從值輸出列表中提取值的名稱
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc，定義值的映射關係
        value = sum(value * value_names.count(name) for name, value in value_map.items()) # 根據映射關係計算總值並返回
        return value
    
    # 以下為 evaluator tree 處理 prompt 的部分
    @staticmethod
    def evaluator_propose_prompt_wrap(x: str, y: str='', eva_y: str='') -> str: # generator
        eva_propose_prompt = generator_of_evaluator_prompt.format(x = x, y = y, eva_y = eva_y)
        print(f'eva_propose_prompt: {eva_propose_prompt}\n')
        return eva_propose_prompt
    
    @staticmethod
    def evaluator_value_prompt_wrap(x: str, y: str='', eva_y: str='') -> str: # evaluator 問 gpt 前的準備
        eva_evaluator_prompt = evaluator_of_evaluator_prompt.format(x = x, y = y, eva_y = eva_y)
        print(f'eva_evaluator_prompt: {eva_evaluator_prompt}\n')
        return eva_evaluator_prompt
    
    @staticmethod
    def evaluator_value_outputs_unwrap(eva_y: str, eva_value_outputs: list) -> float: # evaluator 結束後將文字轉化成 eva_value (該節點的值)
        if eva_y == '': # 沒生出東西算 0 分
            return 0
        eva_value_names = [_.split('\n')[-1] for _ in eva_value_outputs] # 從值輸出列表中提取值的名稱
        eva_value_map = {'wrong': 0.001, 'maybe': 1, 'true': 20}  # TODO: ad hoc，定義值的映射關係
        eva_value = sum(eva_value * eva_value_names.count(eva_name) for eva_name, eva_value in eva_value_map.items()) # 根據映射關係計算總值並返回
        print(f'eva_value: {eva_value}\n')
        return eva_value