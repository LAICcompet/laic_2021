import sys
import re


# get all money in str
def get_all_money(str):
    rule = '([0-9]{1,10}(\.[0-9]{0,4})?(多|余)?(万)?(多|余)?元)'
    match = re.findall(rule,str)
    if match:
        return match
    else:
        return ''


def main():
    calc_money = []
    for line in open(input_path, 'r', encoding='utf-8'):
        line_json = eval(line)
        all_money = get_all_money(line_json['justice'])
        if all_money:
            money_count = 0.00
            for the_money in all_money:
                change_money = the_money[0].replace("多", "").replace("余", "")
                if "万元" in change_money:
                    money_count += float(change_money.replace('万元', '')) * 10000
                elif "千元" in change_money:
                    money_count += float(change_money.replace('千元', '')) * 1000
                else:
                    money_count += float(change_money.replace('元', ''))
            calc_money.append(money_count)
    with open(save_path, "w", encoding="utf-8") as fw:
        for money in calc_money:
            fw.write(str(money)+"\n")


if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    main()