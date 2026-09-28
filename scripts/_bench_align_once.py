"""Benchmark UIE vs full _extract_align for one query."""

import time



from b2c_b2b_bridge import b2b_uie_extract



QUERY = "昨天晚上睡觉属实是太晚了 心脏突突跳动 现在很慌感觉"





def main() -> None:

    from algorithm.chat_preprocess import _extract_align

    from algorithm.companion_cypher import list_align_catalogs



    t0 = time.perf_counter()

    catalogs = list_align_catalogs()

    cat_ms = int((time.perf_counter() - t0) * 1000)



    uie = b2b_uie_extract()

    t1 = time.perf_counter()

    uie.get_uie_predictor()

    load_ms = int((time.perf_counter() - t1) * 1000)



    t2 = time.perf_counter()

    uie.extract_entity(QUERY, ["部位", "疾病", "症状"])

    uie_ms = int((time.perf_counter() - t2) * 1000)



    errors: dict[str, str] = {}

    t3 = time.perf_counter()

    mentions, negated, neg_spans, alignments = _extract_align(QUERY, catalogs, errors)

    align_total_ms = int((time.perf_counter() - t3) * 1000)



    print("query:", QUERY)

    print("catalog_ms", cat_ms)

    print("uie_load_ms", load_ms)

    print("uie_infer_ms", uie_ms)

    print("extract_align_ms", align_total_ms)

    print("mentions", mentions)

    print("alignments", len(alignments))

    for a in alignments:

        print(" ", a.get("mention"), a.get("method"), a.get("standard_name"))





if __name__ == "__main__":

    main()

