from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast
from tokenization import FewVLMTokenizerFast

project_dir = Path(__file__).resolve().parent.parent  
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
flickr_dir = dataset_dir.joinpath('flickr30k')
vg_dir = dataset_dir.joinpath('VG')
flickr_img_dir = flickr_dir.joinpath('images/')
flickr_feature_dir = flickr_dir.joinpath('features')


class COCOCaptionFineTuneDataset(Dataset):
    def __init__(self, split='karpathy_train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)


        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                self.tokenizer = FewVLMTokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)

        data_info_path = dataset_dir.joinpath(f'flickr30k/{args.caption_data}.json')
        with open(data_info_path) as f:
            karpathy_data = json.load(f)

        split_rename = {
            'train': 'train',
            'restval': 'train',
            'val': 'val',
            'test': 'test'
        }

        n_images = 0

        data = []
        for datum in karpathy_data['images']:
            re_split = split_rename[datum['split']]
            if re_split != self.source.split('_')[-1]:
                continue

            if re_split == 'train':
                for d in datum['sentences']:
                    img_id = datum['filename'].split('.')[0]
                    new_datum = {
                        'img_id': img_id,
                        'sent': d['raw'].strip(),
                        'targets': [d['raw'].strip() for d in datum['sentences']],
                        'is_train': True,
                    }
                    data.append(new_datum)
            else:
                img_id = datum['filename'].split('.')[0]
                new_datum = {
                    'img_id': img_id,
                    # 'sent': d['raw'],
                    'targets': [d['raw'].strip() for d in datum['sentences']],
                    'is_train': False,
                }
                data.append(new_datum)

            n_images += 1

        if self.verbose:
            print(f"{self.source} has {n_images} images")
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data
        if args.subsample and 'train' in split:
            random.seed(args.dataseed)
            random.shuffle(self.data)
            if 'train' in split and mode == 'train':
                self.data = self.data[:args.num_data]
            elif 'train' in split and mode == 'val':
                self.data = self.data[args.num_data:2*args.num_data]

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.source_to_h5 = {}

        if self.args.max_n_boxes == 36:
            self.source_to_h5.update({
                'all': flickr_dir.joinpath('features').joinpath('flickr30k_boxes36.h5'),
            })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id



            f = self.source_to_h5['all']

            if isinstance(f, Path):
                # path = self.data_source_to_h5_path[source]
                f = h5py.File(f, 'r')
                # self.split_to_h5_features[split_i] = f
                self.source_to_h5['all'] = f

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            n_boxes = len(boxes)

            feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats = torch.from_numpy(feats)

            if self.args.n_boxes == 100:
                assert n_boxes == 100
                assert len(feats) == 100
                assert len(boxes) == 100

            n_boxes = min(n_boxes, self.args.max_n_boxes)
            out_dict['n_boxes'] = n_boxes
            boxes = boxes[:n_boxes]
            feats = feats[:n_boxes]
            out_dict['boxes'] = boxes
            out_dict['vis_feats'] = feats

        ###### Text #####
        if self.args.no_prefix:
            input_text = ''
            input_ids = []

        else:
            if self.args.prefix is None:
                prefix = ''
            elif self.args.prefix == 'picture':
                prefix = 'a picture of'
            elif self.args.prefix == 'image':
                prefix = 'an image of'
            elif self.args.prefix == 'photo':
                prefix = 'a photo of'



            input_tokens = [prefix]

            input_text = ' '.join(input_tokens)

            if 't5' in self.args.tokenizer:
                input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)

        out_dict['input_text'] = input_text

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        if datum['is_train']:
            sent = datum['sent'].strip()
            if 't5' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(sent, max_length=self.args.gen_max_length, truncation=True)
                # target_ids = self.tokenizer.encode('<extra_id_0> '+sent, max_length=self.args.gen_max_length, truncation=True)

            assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
            out_dict['sent'] = sent
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

        if 'targets' in datum:
            out_dict['targets'] = datum['targets']


        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if self.args.no_prefix:
            assert input_ids.size() == (B, 0)

        if self.args.use_vision:
            V_L = max(entry['n_boxes'] for entry in batch)
            # V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # sentences = []

        targets = []
        img_ids = []
        img_paths = []
        input_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if self.args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i, :n_boxes] = entry['boxes']
                vis_feats[i, :n_boxes] = entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            # sentences.append(entry['sent'])

            if 'targets' in entry:
                targets.append(entry['targets'])


        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

        if self.args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img_id'] = img_ids
            batch_entry['img_paths'] = img_paths

        # batch_entry['sent'] = sentences

        batch_entry['input_text'] = input_text

        batch_entry['targets'] = targets

        batch_entry['task'] = 'caption'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    # if 'mscoco' in split:
    verbose = (gpu == 0)

    dataset = COCOCaptionFineTuneDataset(
        split,
        # raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)
    # elif 'CC' in split:
    #     dataset = CCDataset(split, transform=transform, topk=topk)

    if distributed and mode == 'train':
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        train_sampler = DistributedSampler(dataset)
        # train_sampler = RandomNonreplacmentSampler(dataset, dataset.n_iter)
    else:
        train_sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = COCOCaptionEvaluator()

    loader.task = 'caption'

    return loader



class COCOCaptionEvaluator:
    def __init__(self):
        import language_evaluation
        self.evaluator = language_evaluation.CocoEvaluator(verbose=False)


    def evaluate(self, predicts, answers):

        results = self.evaluator.run_evaluation(predicts, answers)

        return results