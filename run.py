import os
import json
import argparse

from src.tot.tasks import get_task
from src.tot.methods.bfs import solve, naive_solve
#from src.tot.models import gpt_usage

def run(args):
    task = get_task(args.task) # 選擇24，writing，或 crosswords，沒選則默認 game24
    print(f'Task: {args.task}\n')
    logs, eva_logs, cnt_avg, cnt_any = [], [], 0, 0 # 初始化原本 tree、evaluator 的 tree、平均計數、任意計數
    if args.naive_run: # 選擇是否要用 naive run，沒選默認不要
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True) # 創建存原本 tree 的檔案

    for i in range(args.task_start_index, args.task_end_index): # range 決定做的題數
        # solve
        if args.naive_run: 
            ys, info = naive_solve(args, task, i) # 用 naive run
        else:
            ys, info, eva_info = solve(args, task, i) # 用 bfs
        # log infos
        infos = [task.test_output(i, y) for y in ys] # 得到測試結果
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)}) # 添加其他資訊到原本 tree 的資訊列表中
        logs.append(info) # 將 info 添加到 logs 準備寫入檔案
        print(f'ys: {ys}\ninfo: {info} eva_info: {eva_info}\nusage_so_far: {gpt_usage(args.backend)}\n')
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4) # 將整題資訊寫入 json 檔
        # log eva_infos
        # 建立 evaluator tree 的檔案(每題一個檔案，每一個節點都有一棵樹)
        file_eva = f'./logs/eva/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_question{i}.json'
        os.makedirs(os.path.dirname(file_eva), exist_ok=True)
        eva_info.update({'idx': i,}) # 添加題號到 evaluator tree 的 info 字典中
        eva_logs.append(eva_info) # 將 eva_info 添加到 eva_logs 準備寫入檔案
        with open(file_eva, 'w') as f: # 將整題每一棵樹寫入 json 檔
            json.dump(eva_logs, f, indent=4)
        # log main metric
        accs = [info['r'] for info in infos] # 提取每個候選答案的測試結果
        cnt_avg += sum(accs) / len(accs) # 計算平均正確率
        cnt_any += any(accs) # 計算是否有任意正確的答案
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n') # 印出指標信息
    
    n = args.task_end_index - args.task_start_index # 計算任務數量
    print(f'cnt_avg / n, cnt_any / n: {cnt_avg / n, cnt_any / n}\n') # 印出平均正確率和是否有任意正確的答案的比例


def parse_args(): #參數調整
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'openhermes'], default='openhermes')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--task', type=str, choices=['game24', 'text', 'crosswords'], default = 'game24')
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'], default = 'standard')  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'], default = 'sample')
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote', 'tree'], default = 'value')
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='sample')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('')
    print(f'args: {args}\n')
    run(args)