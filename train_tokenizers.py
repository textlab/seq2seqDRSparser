import sys
import os
import json
from importlib import import_module

#from tokenizers import BertWordPieceTokenizer
#from tokenizers.processors import BertProcessing
tokenizer=None
def train_tokenizer(config):
    global tokenizer
    """
    Train a BertWordPieceTokenizer with the specified params and save it
    """

    # Get tokenization params
    save_location = config["out_dir"]
    filename = config["tokenizer_trainer_file"]
    max_length = config["max_length"]
    min_freq = config["min_freq"]
    vocabsize = config["vocab_size"]
    special_tokens = config["special_tokens"]
    lowercase=True
    if 'lowercase' in config:
        if config["lowercase"] == "False":
            lowercase=False
    training_script = "\n".join(config["training_script"])
    exec(training_script,globals(),locals())
    print("Saving tokenizer ...")
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    tokenizer.model.save(save_location)
    f=open(save_location+"/comment.txt","w")
    f.write(config["comment"])
    f.close()

# Identify the config to use
if len(sys.argv) < 2:
    print("No config file specified. Using the default config.")
    configfile = "config.json"
else:
    configfile = sys.argv[1]

# Read the params
with open(configfile, "r") as f:
    config = json.load(f)

tokenizer_out_dir=config["tokenizer_out_dir"]
for tokenizer_dir in config["tokenizers"]:
    if "no_operation" in config["tokenizers"][tokenizer_dir] and config["tokenizers"][tokenizer_dir]["no_operation"] == "True":
        print ("No operation for tokenizer config: "+ tokenizer_dir)
        continue
    tokenizer_config = config["tokenizers"][tokenizer_dir]
    tokenizer_config["out_dir"]=tokenizer_out_dir+"/"+tokenizer_config["out_dir"]
    tokenizer_config["tokenizer_trainer_file"] = config["ml_dataset_out_dir"] +"/"+ tokenizer_config["tokenizer_trainer_dataset"] +"/"+ tokenizer_config["tokenizer_trainer_file"] 
    print("Training tokenizer " + tokenizer_dir)
    print("Tokenizer output_dir:" + tokenizer_config["out_dir"])
    train_tokenizer(tokenizer_config)

