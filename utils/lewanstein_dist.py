from Levenshtein import distance as lev
import json


def main():
    files = ["pit37_v1", "umowa_sprzedazy_samochodu",
             "umowa_o_dzielo",
             "umowa_na_odleglosc_odstapienie",
             "pozwolenie_uzytkowanie_obiektu_budowlanego"]
    for name in files:
        with open(f"{name}.json") as file:
            data = json.loads(file.read())
        used = set()
        app_new = dict()
        for x in data:
            for y in data:
                if (lev(x[0], y[0]) < 2) and (x != y) and (y[0] not in used) and (x[0] not in used):
                    used.add(y[0])
                    if x[0] not in app_new.keys():
                        app_new[x[0]] = x[1]
                    app_new[x[0]] += y[1]
                    print(f"Match between: [{x[0]}] and [{y[0]}]")

        new = sorted(app_new.items(), key=lambda z: z[1], reverse=True)
        tmp = dict()
        for key, val in new[:30]:
            tmp[key] = val

        with open(f"{name}_top.json", "w") as file:
            test = json.dumps(tmp, indent=4)
            file.write(test)


if __name__ == '__main__':
    print(lev("wykonawey", "wykonawea"))
    main()
