"""One-off UIE load + infer timing. Run from B2C business root."""

import time



from b2c_b2b_bridge import b2b_uie_extract



QUERY = "头部常见有哪些病"

SCHEMA = ["部位", "疾病", "症状"]





def main() -> None:

    uie = b2b_uie_extract()

    t0 = time.perf_counter()

    uie.get_uie_predictor()

    t1 = time.perf_counter()

    print("load_s", round(t1 - t0, 2))



    for i in range(3):

        t2 = time.perf_counter()

        out = uie.extract_entity(QUERY, SCHEMA)

        ms = int((time.perf_counter() - t2) * 1000)

        print(f"infer_{i+1}_ms", ms, "keys", list(out.keys()) if out else out)





if __name__ == "__main__":

    main()

