import tensorflow as tf

from mt.config import config

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from bs4 import BeautifulSoup
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import seaborn as sns


def get_image_from_otto(pid):
    base_url = "https://www.otto.de/p/"
    url = base_url + pid
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    image_url = soup.find('meta', property="og:image")

    if image_url:
        r = requests.get(image_url["content"], headers={'User-agent': 'Mozilla/5.0'})
        return Image.open(BytesIO(r.content))


def print_tensor_w_text(tensor, prefix = "DOCUMENT", scores=None):
    if isinstance(tensor, tf.RaggedTensor):
        tensor = tensor.to_tensor()
    if tf.rank(tensor) > 1:
        for num, i in enumerate(tensor):
            score = scores[num].numpy() if scores is not None else ""
            print(f"{prefix} {num}: ", " ".join([x.decode("utf-8") for x in i.numpy() if x != b"[PAD]"]), score)
    else:
        print(prefix, " ".join([x.decode("utf-8") for x in tensor.numpy() if x != b"[PAD]"]))


def eval_validation_by_inspection(val, model, tokenizer, show_images=False, w_attn=False, score_weights=None):

    # val = next(iter(ds_eval))
    batch_id = np.random.choice(list(range(val[config.QUERY_COL].shape[0])), 1)[0]

    mask = tf.not_equal(val[config.CLICK_COL][batch_id], -1)

    query = tokenizer.detokenize(val[config.QUERY_COL])[batch_id]
   
    
    clicked_docs = tokenizer.detokenize(
        tf.boolean_mask(val["product_pd_Name"][batch_id], tf.equal(val["click"][batch_id], 1)))

    num_prods = tf.boolean_mask(val["product_pd_Name"][batch_id], mask).shape[0]

    print_tensor_w_text(query, "QUERY")

    if show_images:
        
        model = [model] if not isinstance(model, list) else model
        imgs = []
        for m in model:
            pred = m.predict(val)
            pred = tf.boolean_mask(pred[batch_id], mask)
            pred = tf.squeeze(pred)
            ranks = tf.argsort(pred, direction="DESCENDING")
            ranked_ids = tf.gather(val["offer_or_product_id"][batch_id], ranks)

            for pid in ranked_ids.numpy():
                img = get_image_from_otto(pid.decode("utf-8"))
                imgs.append(img)
        
        fig = plt.figure(figsize=(10. * len(model), 5. * num_prods))
        grid = ImageGrid(
            fig, 111, 
            nrows_ncols=(num_prods, len(model)),
            axes_pad=0.1,
            direction="column")

        curr_model_id = 0
        for ax, (num, im) in zip(grid, enumerate(imgs)):
            if im:
                if num % num_prods == 0:
                    ax.set_title(model[curr_model_id].name)
                    curr_model_id += 1
                ax.imshow(im, aspect='auto')
    else:
        if w_attn:
            attn_mask = tf.not_equal(tf.concat((tf.ones((1,), dtype=tf.int64), val[config.QUERY_COL][batch_id]), axis=0), 0)
            pred, attn = model(val, return_attention=True, training=False)
            attn = attn[batch_id]
            print("ATTENTION", tf.boolean_mask(tf.squeeze(attn), attn_mask))
        else:
            pred = model.predict(val)
        
        if isinstance(pred, list):
            if score_weights is None:
                score_weights = [1] * len(pred)
            combined_pred = tf.zeros_like(pred[0], dtype=tf.float32)
            for i,p in enumerate(pred):
                combined_pred = tf.add(combined_pred, tf.multiply(tf.cast(score_weights[i], tf.float32), p))
            pred = combined_pred

        pred = tf.boolean_mask(pred[batch_id], mask)

        docs = tf.boolean_mask(tokenizer.detokenize(val["product_pd_Name"][batch_id]).to_tensor(), mask)
        ids = tf.expand_dims(tf.boolean_mask(val["offer_or_product_id"][batch_id], mask), axis=1)
        docs = tf.concat((ids, docs), axis=1)
        ranks = tf.argsort(pred, direction="DESCENDING")
        ranked_docs = tf.gather(docs, ranks)
        ranked_pred = tf.gather(pred, ranks)    
        print('─' * 100)
        print_tensor_w_text(clicked_docs, "CLICKED DOCUMENTS")
        print('─' * 100)
        print_tensor_w_text(ranked_docs, "RANKED DOCS", ranked_pred)


def prepare_tokens_for_print(token_hashes, tokenizer):
    if tf.equal(tf.rank(token_hashes), 1):
        token_hashes = tf.expand_dims(token_hashes, 0)
    if tf.equal(tf.rank(token_hashes), 3):
        token_hashes = tf.squeeze(token_hashes, 0)
    tokens = tokenizer.detokenize(token_hashes).to_tensor().numpy()
    print_tokens = []
    for token in tokens:
        token_string = " ".join([x.decode("utf-8") for x in token if x != b"[PAD]"])
        print_tokens.append(token_string)
    return print_tokens
    

def print_scores_pointwise(data, model, tokenizer, limit=20):

    q = tokenizer.detokenize(data["searchterm"]).to_tensor().numpy()
    d = tokenizer.detokenize(data["product_pd_Name"]).to_tensor().numpy()
    r = model.predict(data)
    r = r.numpy()

    c = data["click"].numpy()
    
    cnt = 0
    for query, doc, rel, click in zip(q,d,r, c):
        print("QUERY: ", " ".join([x.decode("utf-8") for x in query if x != b"[PAD]"]))
        print("DOC: ", " ".join([x.decode("utf-8") for x in doc if x != b"[PAD]"]))
        print("REL: ", rel)
        print("CLICK: ", click)
        print('─' * 100)
        cnt += 1
        if cnt >= limit:
            break
            
def print_scores_pairwise(data, model, tokenizer, limit=20):
    q = tokenizer.detokenize(data["searchterm"]).to_tensor().numpy()
    d = tokenizer.detokenize(data["product_pd_Name"]).to_tensor().numpy()
    r = model.predict(data)
    r = r.numpy()

    c = data["click"].numpy()

    cnt = 0
    for query, docs, rel, click in zip(q,d,r, c):
        print("QUERY: ", " ".join([x.decode("utf-8") for x in query if x != b"[PAD]"]))
        for _, doc in enumerate(docs):
            print("DOC: ", " ".join([x.decode("utf-8") for x in doc if x != b"[PAD]"]))
            print("REL: ", rel[_])
            print("CLICK: ", click[_])
        print('─' * 100)
        cnt += 1
        if cnt >= limit:
            break

def make_pb_plot(trainer, layout_encoder, devices=["mobile", "desktop"], layouts=["galleryQuad", "list"]):
    fig = plt.figure(figsize=(15,10))
    for device in devices:
        for layout in layouts:
            input_ = {
                "actual_position": tf.expand_dims(tf.range(1, config.MAX_SEQ_LENGTH+1, dtype=tf.int64), 0),
                "device": tf.expand_dims(tf.repeat(device, [config.MAX_SEQ_LENGTH], axis=0), 0),
                "layout_type": tf.expand_dims(tf.repeat(layout, [config.MAX_SEQ_LENGTH], axis=0), 0)
            }
            props = trainer.propensity_estimator(layout_encoder(input_)["pb_features"])
            plt.plot(tf.squeeze(props), label=device+" "+layout)
    for i in range(0,72,2):
        plt.axvline(i, alpha=.3)
    plt.legend()
    return fig, props


def propensities_to_pandas(trainer, layout_encoder):
    inputs = []
    for device in layout_encoder.devices:
        for layout in layout_encoder.layout_types:
            input_ = {
                "actual_position": tf.expand_dims(tf.range(1, config.MAX_SEQ_LENGTH+1, dtype=tf.int64), 0),
                "device": tf.expand_dims(tf.repeat(device, [config.MAX_SEQ_LENGTH], axis=0), 0),
                "layout_type": tf.expand_dims(tf.repeat(layout, [config.MAX_SEQ_LENGTH], axis=0), 0)
            }
            props = trainer.propensity_estimator(layout_encoder(input_)["pb_features"])
            input_["propensity"] = props
            for k,v in input_.items():
                input_[k] = tf.squeeze(v).numpy()
            inputs.append(input_)
    df = pd.concat([pd.DataFrame.from_dict(inputs[i]) for i in range(len(inputs))], axis=0, ignore_index=True)
    return df


def sms_vs_tfidf_plot(pipe, ranker, inputs, idx, limit=10):
    from mt.models.layers import SemanticMatchingScorer, PbkClassification
    df = pd.DataFrame(inputs["num_features"][idx].numpy(), 
                  columns= [f"{k}_{i}" for k,v in config.CATEGORICAL_FEATURES.items() for i in range(len(v))] + config.NUMERICAL_COLUMNS)

    sms = [x for x in ranker.layers if isinstance(x, SemanticMatchingScorer)][0](inputs).numpy().squeeze(-1)
    pbk_match = [x for x in ranker.layers if isinstance(x, PbkClassification)][0](inputs).numpy().squeeze(-1)
    df["sms"] = sms[idx]
    df["pbk_match"] = pbk_match[idx]

    titles = []
    for t in pipe.tokenizer.detokenize(inputs[config.PRODUCT_TITLE_COL][idx]).to_tensor().numpy():
        t = " ".join([x.decode("utf-8") for x in t if x != b"[PAD]"])
        titles.append(t)
    # print(titles)
    query =  pipe.tokenizer.detokenize(inputs[config.QUERY_COL])[idx].numpy()
    query = " ".join([x.decode("utf-8") for x in query if x != b"[PAD]"])

    df["click_score"] = ranker(inputs)[0][idx]
    df["order_score"] = ranker(inputs)[1][idx]

    # df = (df - df.min(axis=0))/(df.max(axis=0) - df.min(axis=0) + 1e-9)

    plt.figure(figsize=(16,6 * (limit/10)))
    # plotte hier als x axis ticks die produkt-titel und nehme nur die spalten tfidf / bm25 und cos distanz um zu zeigen, dass die cos 
    # distanz viel Geiler ist
    ax = sns.heatmap(df.iloc[:limit], annot=True)
    ax.set_yticks(np.arange(limit)+0.4)
    ax.set_yticklabels(titles[:limit], rotation = 0)
    # plt.yticks(rotation=90)
    ax.set_title(f"Query: {query}")
    plt.savefig("tfidf_vs_sms.svg")