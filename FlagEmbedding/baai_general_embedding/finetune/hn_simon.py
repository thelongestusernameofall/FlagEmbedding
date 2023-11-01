import argparse
import json
import random

import faiss
from tqdm import tqdm

from FlagEmbedding import FlagModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-base-en", type=str)
    parser.add_argument('--input_file', default=None, type=str, required=True, help="action_dict file path")
    parser.add_argument('--candidate_pool', default=None, type=str)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--range_for_sampling', default=None, type=str, help="range to sample negatives")
    parser.add_argument('--use_gpu_for_searching', action='store_true', help='use faiss-gpu')
    parser.add_argument('--negative_number', default=15, type=int, help='negative number for each query')
    parser.add_argument('--query_instruction_for_retrieval', default="")
    parser.add_argument('--batch_size', default=256, type=int, help='batch size for inference')

    return parser.parse_args()


def create_index(embeddings, use_gpu):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(index,
                 query,
                 topk: int = 200,
                 batch_size: int = 64):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(batch_query, k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def get_corpus(candidate_pool):
    corpus = []
    for line in open(candidate_pool):
        line = json.loads(line.strip())
        corpus.append(line['text'])
    return corpus


def get_prefix():
    prefix_list = [
        "请不要介意我提问",
        "我希望您能协助我分析",
        "如果您能抽出时间",
        "在这个问题上",
        "非常感谢您的支持",
        "我想请教您",
        "我想请您帮我看看",
        "我想请您帮我分析一下",
        "我想请您帮我分析一下这个问题",
        "如果您可以的话",
        "如果您有时间的话",
        "如果您有时间的话，我想请您帮我分析一下这个问题",
        "如果您不麻烦的话",
        "我对这个问题很感兴趣",
        "我需要更多信息",
        "如果您方便的话",
        "我疑惑不解",
        "感谢您的耐心",
        "我希望了解更多"
    ]
    # random sample one prefix
    sample = random.sample(prefix_list, 1)[0]
    # if contain "一下" 30% chance to change to "一下子" 30% "下"
    if "一下" in sample:
        if random.random() < 0.3:
            sample = sample.replace("一下", "一下子")
        elif random.random() < 0.3:
            sample = sample.replace("一下", "下")

    # if containt "如果您" 30% chance to change to "如果"， 30% remove it
    if "如果您" in sample:
        if random.random() < 0.3:
            sample = sample.replace("如果您", "如果")

    if "如果" in sample:
        if random.random() < 0.3:
            sample = sample.replace("如果", "")

    # if contain "您" 50% chance to change to "你":
    if "您" in sample:
        if random.random() < 0.5:
            sample = sample.replace("您", "你")

    return sample


def get_suffix():
    suffix_list = [
        "我会耐心等待您的消息",
        "如果能提供更多细节，将不胜感激",
        "如果您愿意提供帮助，我将非常感激",
        "如果有任何问题，请随时联系我",
        "感谢您的协助",
        "如果需要更多细节，请告诉我",
        "如果有补充说明，请随时补充",
        "如果有任何更新，请通知我",
        "如果有进一步的建议，请不吝告知",
        "期待听到您的观点",
        "如果可以的话，请告知",
        "如果有任何更新，请通知我",
    ]
    # random sample one suffix
    sample = random.sample(suffix_list, 1)[0]
    # if contain "如果" 50% chance change to "若"
    if "如果" in sample:
        if random.random() < 0.50:
            sample = sample.replace("如果", "若")
    # 50% chance change '我' to '我们'
    if "我" in sample:
        if random.random() < 0.50:
            sample = sample.replace("我", "我们")
    # 50% chance change '您' to '你'
    if "您" in sample:
        if random.random() < 0.50:
            sample = sample.replace("您", "你")

    return sample


def create_querys(query, number=5):
    result = []
    for i in range(number):
        item = get_prefix() + "," + query + "," + get_suffix() + "."
        # 50% chance to remove 【】
        if "【" and "】" in item and random.random() < 0.5:
            item = item.replace("【", "").replace("】", "")
        result.append(item)
    return result


def find_knn_neg_old(model, input_file, candidate_pool, output_file, sample_range, negative_number, use_gpu,
                     batch_size=256):
    corpus = []
    queries = []
    train_data = []
    # action_dict = {}

    fh = open(input_file, "r", encoding="utf-8")
    action_dict = json.loads(fh.read())
    fh.close()

    prompt_dict = {}
    # prompts to action map
    for action, prompts in action_dict.items():
        for prompt in prompts:
            prompt_dict[prompt] = action
            corpus.append(prompt)
            # queries.extend(create_querys(prompt, number=1))

    corpus = list(set(corpus))
    for prompt in corpus:
        queries.extend(create_querys(prompt, number=1))

    pool_corpus = []
    if candidate_pool is not None:
        pool_corpus = get_corpus(candidate_pool)
        pool_corpus = list(set(pool_corpus))
    corpus.extend(pool_corpus)

    print(f'inferencing embedding for corpus (number={len(corpus)})--------------')
    p_vecs = model.encode(corpus, batch_size=batch_size)

    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode_queries(queries, batch_size=batch_size)

    print('creat index and search------------------')
    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1])

    for i, query in enumerate(queries):
        prompt = corpus[i]
        action = prompt_dict[prompt]
        item = {"query": query, "pos": [prompt]}
        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1: break
            if corpus[inx] not in item['pos'] and corpus[inx] != query and corpus[inx] not in prompt_dict or \
                    prompt_dict[corpus[inx]] != action:
                filtered_inx.append(inx)
        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)
        item['neg'] = [corpus[inx] for inx in filtered_inx]
        train_data.append(item)

    with open(output_file, 'w') as f:
        for data in train_data:
            # if len(data['neg']) < negative_number:
            #     data['neg'].extend(random.sample(corpus, negative_number - len(data['neg'])))
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def find_knn_neg(model, input_file, candidate_pool, output_file, sample_range, negative_number, use_gpu,
                 batch_size=256):
    corpus = []
    queries = []
    poses = {}
    train_data = []
    # action_dict = {}

    fh = open(input_file, "r", encoding="utf-8")
    action_dict = json.loads(fh.read())
    fh.close()

    prompt_dict = {}
    # prompts to action map
    for action, prompts in action_dict.items():
        for prompt_pair in prompts:
            for prompt, pos in prompt_pair.items():
                prompt_dict[prompt] = action
                poses[prompt] = pos

                corpus.append(prompt)
                queries.append(prompt)
                corpus.append(pos)

    corpus = list(set(corpus))
    # for prompt in corpus:
    #     queries.extend(create_querys(prompt, number=1))

    pool_corpus = []
    if candidate_pool is not None:
        pool_corpus = get_corpus(candidate_pool)
        pool_corpus = list(set(pool_corpus))
    corpus.extend(pool_corpus)

    print(f'inferencing embedding for corpus (number={len(corpus)})--------------')
    p_vecs = model.encode(corpus, batch_size=batch_size)

    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode_queries(queries, batch_size=batch_size)

    print('creat index and search------------------')
    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1])

    for i, query in enumerate(queries):
        pos = poses[query]
        action = prompt_dict[query]

        item = {"query": query, "pos": [pos]}
        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1: break
            # corpus[inx] not in item['pos'] and corpus[inx] != query and
            if corpus[inx] not in prompt_dict or prompt_dict[corpus[inx]] != action:
                filtered_inx.append(inx)
        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)
        item['neg'] = [corpus[inx] for inx in filtered_inx]
        train_data.append(item)

    with open(output_file, 'w') as f:
        for data in train_data:
            # if len(data['neg']) < negative_number:
            #     data['neg'].extend(random.sample(corpus, negative_number - len(data['neg'])))
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = get_args()
    sample_range = args.range_for_sampling.split('-')
    sample_range = [int(x) for x in sample_range]

    model = FlagModel(args.model_name_or_path, query_instruction_for_retrieval=args.query_instruction_for_retrieval)

    find_knn_neg(model,
                 input_file=args.input_file,
                 candidate_pool=args.candidate_pool,
                 output_file=args.output_file,
                 sample_range=sample_range,
                 negative_number=args.negative_number,
                 use_gpu=args.use_gpu_for_searching,
                 batch_size=args.batch_size)
