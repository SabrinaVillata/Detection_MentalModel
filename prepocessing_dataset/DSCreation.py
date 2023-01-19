import json
import csv
import glob

if __name__ == '__main__':
    # read all json file
    list_of_files = sorted(glob.glob('train/*.json'))
    for file_name in list_of_files:
        print("File name: " + file_name)
        j = open(file_name, 'r')
        data = json.load(j)

        header = ["dialogue_id", "speaker", "turn", "act_1", "act_2", "act_3"]

        ## New CSV file ##
        n_file = file_name[16:19]
        f = open('CSV_train/users_dialog_' + n_file + '.csv', 'w', encoding='UTF8')
        writer = csv.writer(f)
        writer.writerow(header)

        id = ""  # keep track of dialogue id

        for dialog in data:
            for turn in dialog["turns"]:
                act_1 = None
                act_2 = None
                act_3 = None
                # saves different acts of the turn
                act_1 = turn["frames"][0]["actions"][0]["act"]
                len_actions = len(turn["frames"][0]["actions"])
                # check if there are still actions and if they are different from previous
                if len_actions > 1 and turn["frames"][0]["actions"][1]["act"] != None and turn["frames"][0]["actions"][1]["act"] != act_1:
                    act_2 = turn["frames"][0]["actions"][1]["act"]
                # check if there are still actions and if they are different from previous (max of three)
                if len_actions > 2 and turn["frames"][0]["actions"][2]["act"] != None and turn["frames"][0]["actions"][2]["act"] != act_2 and turn["frames"][0]["actions"][2]["act"] != act_1:
                    act_3 = turn["frames"][0]["actions"][2]["act"]
                # create new line in CSV file
                line = [dialog["dialogue_id"], turn["speaker"], turn["utterance"], act_1, act_2, act_3]
                writer.writerow(line)

        j.close()
        f.close()