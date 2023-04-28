import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset
#import datasets
import subprocess
import shutil
from data import TranslationDataset

# Identify the config file
if len(sys.argv) < 2:
    print("No config file specified. Using the default config.")
    configfile = "config.json"
else:
    configfile = sys.argv[1]
print ("Using config file:", configfile)
# Read the params
with open(configfile, "r") as f:
    config = json.load(f)

model_confs = config["models"]

enc_tokenizer=None
dec_tokenizer=None
model=None
criterion=None
optimizer=None
device=None

global encoder_start_token_id
global encoder_bos_token_id
global decoder_start_token_id
global decoder_bos_token_id
encoder_start_token_id=5
encoder_bos_token_id=0
decoder_start_token_id=0
decoder_bos_token_id=0

os.environ["TOKENIZERS_PARALLELISM"]="false"

def compute_loss(predictions, targets):
    """Compute our custom loss"""
    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]

    rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
    rearranged_target = targets.contiguous().view(-1)

    loss = criterion(rearranged_output, rearranged_target)

    return loss

def count_parameters(mdl):
        return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

def load_dataset(en_file, dec_file, en_tokenizer, dec_tokenizer, enc_maxlength, dec_maxlength, batch_size ):
    dataset=TranslationDataset(en_file, dec_file, en_tokenizer, dec_tokenizer, enc_maxlength, dec_maxlength)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, \
                                                drop_last=True, num_workers=1, collate_fn=dataset.collate_function)
    return (dataset, dataloader)

def load_tokenizer(tokenizer_id):
    global config
    global tokenizer
    loading_script = "\n".join(config["tokenizers"][tokenizer_id]["loading_script"])
    loading_script=loading_script.replace("{out_dir}", "'" + config["tokenizer_out_dir"] + "/" + config["tokenizers"][tokenizer_id]["out_dir"] + "'").replace("{lowercase}", config["tokenizers"][tokenizer_id]["lowercase"])
    exec(loading_script,globals(),locals())
    #if "special_tokens" in config["tokenizers"][tokenizer_id] and type(config["tokenizers"][tokenizer_id]["special_tokens"]) == list:
    #        tokenizer.add_special_tokens({'additional_special_tokens':config["tokenizers"][tokenizer_id]["special_tokens"]})

    return tokenizer


def train_model (model, train_dataloader, device, optimizer, num_train_batches):
    model.train()
    epoch_loss = 0

    for i, (en_input, en_masks, de_output, de_masks) in enumerate(train_dataloader):

        optimizer.zero_grad()

        en_input = en_input.to(device)
        de_output = de_output.to(device)
        en_masks = en_masks.to(device)
        de_masks = de_masks.to(device)

        lm_labels = de_output.clone()
        out = model(input_ids=en_input, attention_mask=en_masks,
                decoder_input_ids=de_output, decoder_attention_mask=de_masks,labels=lm_labels)
        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, de_output)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
    m_o_l = epoch_loss / num_train_batches
    print("Mean epoch loss:", m_o_l)
    return m_o_l



def quotation_process(s):
    in_quota=False
    new_string = ''
    for i in range(len(s)):
        if s[i]=='"':
            if in_quota:
                new_string+='"'
                in_quota =  False
            else:
                in_quota = True
                new_string+='"'
        elif s[i]==' ':
            if not in_quota:
                new_string+= ' '
        else:
            new_string+=s[i]
    return new_string



def fix_line(s):

    # Fix a few obvious problems
    s=s.replace("$NEWDRS", "$NEW DRS")
    s=s.replace("$NEWREF", "$NEW REF")

    # Fix problem of  " ~ "
    s=s.replace(" ~ ", "~")

    # Fix problem of Co -Something:
    s=s.replace(" -", "-")

    # Fix problem of somehting _ something:
    s=s.replace(" _ ", "_")

    # Fix problem of " ~"
    s=s.replace(" ~","~")

    # Fix orphan \" in the end
    if s.count("\"") %2 ==1 and s.endswith(" \""):
        s=s[:-2]

    # Fix problem of something"n.01 " :
    in_quot=False
    previousChar = ""
    updatedStr = ""
    for i in range(len(s)):
        if s[i]=='"' and in_quot==False:
            in_quot=True
            if previousChar != " ":
                updatedStr+=" "
        elif s[i] =='"' and in_quot==True:
            in_quot=False
            if previousChar==" ":
                updatedStr=updatedStr[0:-1]

        updatedStr+=s[i]
        previousChar=s[i]
    s=updatedStr

    if len(s)<3:
        return s

    # Fix $0Name problem in advance to the following operations
    s=s.replace("$0Name", "$0 Name")

    # Fix problem of $numSomething:
    if(s[0]=='$'):
        if(s[1]=='-'):
            updatedStr = "$-"
            i=2
            previousChar = "-"
        else:
            updatedStr = "$"
            i=1
            previousChar = "$"

        while i< len (s):
            if not s[i].isnumeric():
                if s[i] != ' ' and s[i]!='N':
                    updatedStr += " " + s[i:]
                else:
                    updatedStr += s[i:]
                break
            updatedStr+= s[i]
            previousChar = s[i]
            i += 1

        return updatedStr
    return s

def generate_output_from_model_output(inp, model, enc_tokenizer, dec_tokenizer, enc_max_length, device):
    # cut off at BERT max length 512
    inputs = enc_tokenizer(inp, padding="max_length", truncation=True, max_length=enc_max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=enc_max_length)
    output_str = dec_tokenizer.batch_decode(outputs,skip_special_tokens=True, clean_up_tokenization_spaces=False, early_stopping=True)
    # Here we find the first [SEP] in the string. We also do some post-processing such as handling spaces and quotation marks
    for i in range(len(output_str)):
        start = output_str[i].find('|')
        end = output_str[i].find('[SEP]')
        output_str[i]=output_str[i][start+1:end]
        output_str[i]=output_str[i].strip()
        output_str[i]=quotation_process(output_str[i])
        output_str[i]=output_str[i].replace('|','\n').replace('@ ','@').replace('- ', '-').replace('$ ','$').replace('" n', '"n').replace('" v','"v').replace('. ','.')
        lines=output_str[i].split('\n')
        for j in range(len(lines)):
            line = lines[j].replace('$', ' $').replace('@',' @').replace(" \" @", "\" @")
            new_line = line + '  '
            while (new_line!=line):
                line=new_line
                new_line=new_line.replace('  ', ' ')
            line=line.strip()
            lines[j]=line
            line = line.split(' ')
            for k in range(len(line)):
                term = line[k]
                if not len(term)>0:
                    line[k]=""
                    continue
                if term[0]=='"' and not term[len(term)-1]=='"':
                    term=term[1:]
                elif not term[0]=='"' and term[len(term)-1]=='"':
                    term=term[:-1]
                line[k]=term
            line = fix_line(' '.join(line))
            line=line.strip()
            lines[j] = line

        output_str[i] = '\n'.join([line for line in lines if line!="$" ] )
        #output_str[i]=lines#translate_rel2std(quotation_process(output_str[i]))

    return output_str


def create_model(script, enc_vocabsize, enc_max_length, dec_vocabsize, dec_max_length):
    global encoder_start_token_id
    global encoder_bos_token_id
    global decoder_start_token_id
    global decoder_bos_token_id
    global model
    global criterion
    global device
    global optimizer

    if type(script) == list:
        script = "\n".join(script)
    script=script.replace("{enc_vocabsize}", str(enc_vocabsize)).replace("{enc_max_length}",str(enc_max_length)).replace("{dec_vocabsize}",str(dec_vocabsize)).replace("{dec_max_length}",str(dec_max_length))
    exec(script,globals(),locals())
    return (model,criterion, device, optimizer, encoder_start_token_id, encoder_bos_token_id, decoder_start_token_id, decoder_bos_token_id)

def run_shell_command(cmd):
    proc = subprocess.Popen(cmd  , stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    out=proc.communicate()[0].decode()
    return out
def extract_f1(out):
    out=out.split("\n")
    for item in out:
        item = item.split(":")
        if item[0].strip() == "F-score":
            return float(item[1][1:])
    return float(0)

def create_batches(lst, batch_size):
    batches=[]
    now=0
    my_batch=[]
    for item in lst:
        if now <batch_size:
            my_batch.append(item)
            now +=1
        else:
            batches.append(my_batch)
            my_batch=[item]
            now=1
    batches.append(my_batch)
    return batches



def eval_model(model, batch_input, out_file, enc_tokenizer, dec_tokenizer, enc_max_length, device, batch_size, encoder_start_token_id, encoder_bos_token_id, decoder_start_token_id, decoder_bos_token_id, postprocess_command, compare_command):
    model.eval()
    model.config.encoder.decoder_start_token_id=encoder_start_token_id
    model.config.encoder.bos_token_id=encoder_bos_token_id
    model.config.decoder.decoder_start_token_id=decoder_start_token_id
    model.config.decoder.bos_token_id=decoder_bos_token_id

    results = list(map(lambda x: generate_output_from_model_output(x, model, enc_tokenizer, dec_tokenizer, enc_max_length, device), batch_input))

    f=open('temp.txt','w')
    for batch in results:
        for result in batch:
            f.write(result.replace('\n','***'))
            f.write('\n')
    f.close()

    command = postprocess_command.replace("{input_file}", "temp.txt" ).replace("{output_file}","inp.txt")
    print("Running command: ", command)
    out = run_shell_command(command)
    print(out)
    command = compare_command.replace("{test_file}", "inp.txt" ).replace("{gold_file}",out_file)
    print ("Running command: ", command)
    out=run_shell_command(command)
    print(out)
    f1=extract_f1(out)
    return (f1, out)

def save_model_to_location(save_location, model, output):
    print("Saving model to: " + save_location)
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    f=open(save_location + "/output","w")
    f.write(output)
    f.close()
    save_location = os.path.join(save_location, "encdec.mdl")
    torch.save(model, save_location)

def get_batch_dataset(in_file, batch_size):
    data=[]
    f=open(in_file,"r")
    line = f.readline()
    while(line):
        line = line.replace("\n","")
        data.append(line)
        line=f.readline()
    return create_batches(data, batch_size)

for model_conf in model_confs:
    conf=model_confs[model_conf]
    if conf["do_train"]!="True":
        print("Nothing is done for " + model_conf )
        continue

    print("Starting process for training model config: "+ model_conf)
    print("Loading encoder tokenizer " + conf["enc_tokenizer"])
    enc_tokenizer=load_tokenizer(conf["enc_tokenizer"])
    print("Loading decoder tokenizer " + conf["dec_tokenizer"])
    dec_tokenizer=load_tokenizer(conf["dec_tokenizer"])

    print("Loading config:")
    print("Encoder max length: " + str(config["tokenizers"][conf["enc_tokenizer"]]["max_length"]))
    print("Encoder vocabulary size: " + str(config["tokenizers"][conf["enc_tokenizer"]]["vocab_size"]))
    print("Decoder max length: " + str(config["tokenizers"][conf["dec_tokenizer"]]["max_length"]))
    print("Decoder vocabulary size: " + str(config["tokenizers"][conf["dec_tokenizer"]]["vocab_size"]))
    print("Batch size: " + str(conf["batch_size"]))

    print("Loading training dataset training input file:" + conf["train_input_file"] + ", training output file:", conf["train_output_file"])

    (train_dataset, train_dataloader) = load_dataset(conf["train_input_file"], conf["train_output_file"], enc_tokenizer, dec_tokenizer, config["tokenizers"][conf["enc_tokenizer"]]["max_length"], config["tokenizers"][conf["dec_tokenizer"]]["max_length"],conf["batch_size"])

    print("Loading dev dataset input file:" + conf["dev_input_file"] + ", dev dataset output file:", conf["dev_output_file"])
    dev_batch_input=get_batch_dataset(conf["dev_input_file"], conf["batch_size"])

    print("Loading test dataset input file:" + conf["test_input_file"] + ", test dataset output file:", conf["test_output_file"])
    test_batch_input=get_batch_dataset(conf["test_input_file"], conf["batch_size"])

    if("eval_input_file" in conf and "eval_output_file" in conf and conf["eval_input_file"]!="" and conf["eval_output_file"]!=""):
        print("Loading eval dataset input file:" + conf["eval_input_file"] + ", eval dataset output file:", conf["eval_output_file"])
        eval_batch_input=get_batch_dataset(conf["eval_input_file"], conf["batch_size"])

    print("Creating the base model")
    (model, criterion, device, optimizer, encoder_start_token_id, encoder_bos_token_id, decoder_start_token_id, decoder_bos_token_id) = create_model(conf["model_creation_script"],
             config["tokenizers"][conf["enc_tokenizer"]]["vocab_size"],
             config["tokenizers"][conf["enc_tokenizer"]]["max_length"],
             config["tokenizers"][conf["dec_tokenizer"]]["vocab_size"],
             config["tokenizers"][conf["dec_tokenizer"]]["max_length"])

    # Continue training of a model
    if "continue_from_model" in conf and conf["continue_from_model"]!="":
        print("Loading model:", conf["continue_from_model"])
        model = torch.load(conf["continue_from_model"])

    epoch_evals = {}
    epoch_devs={}
    epoch_tests={}
    epoch_losses={}
    max_eval_val = -1
    all_output=""
    for epoch in range(conf['num_epochs']):
        print("Starting epoch", epoch)
        mean_epoch_loss = train_model(model, train_dataloader, device, optimizer, conf["batch_size"])

        epoch_losses[epoch] = mean_epoch_loss

        (epoch_dev, dev_out) = eval_model(model, dev_batch_input, conf["dev_output_file"], enc_tokenizer, dec_tokenizer, config["tokenizers"][conf["enc_tokenizer"]]["max_length"], device, conf["batch_size"], encoder_start_token_id, encoder_bos_token_id, decoder_start_token_id, decoder_bos_token_id, conf["postprocess_exec"], conf["compare_exec"])

        print("Epoch dev F1:",epoch_dev)

        (epoch_test, test_out) = eval_model(model, test_batch_input, conf["test_output_file"], enc_tokenizer, dec_tokenizer, config["tokenizers"][conf["enc_tokenizer"]]["max_length"], device, conf["batch_size"], encoder_start_token_id, encoder_bos_token_id, decoder_start_token_id, decoder_bos_token_id, conf["postprocess_exec"], conf["compare_exec"])

        print("Epoch test F1:",epoch_test)

        epoch_eval = ""
        eval_out = ""
        if(conf["eval_input_file"]!="" and conf["eval_output_file"]!=""):
            (epoch_eval, eval_out)=eval_model(model, eval_batch_input, conf["eval_output_file"], enc_tokenizer, dec_tokenizer, config["tokenizers"][conf["enc_tokenizer"]]["max_length"], device, conf["batch_size"], encoder_start_token_id, encoder_bos_token_id, decoder_start_token_id, decoder_bos_token_id, conf["postprocess_exec"], conf["compare_exec"])
            print("Epoch eval F1:",epoch_eval)
        
        save_location = conf['model_save_path'] + model_conf +"/"+ str(epoch)
        output_of_this_epoch = "Dev:\n" + dev_out + "\nTest:\n" + test_out + "\nEval:\n"+ eval_out + "\nF1s of this epoch: Dev: "+str(epoch_dev) + " ,Test: " + str(epoch_test) +  ",Eval: " + str(epoch_eval)

        epoch_devs[epoch]=epoch_dev
        epoch_evals[epoch]=epoch_eval
        epoch_tests[epoch]=epoch_test

        output_of_this_epoch +="\nHistory: Devs:\n "+ str(epoch_devs) + "\nTests:\n" + str(epoch_tests) + "\nEvals:\n" + str(epoch_evals)

        all_output+="\nEpoch "+ str(epoch) + "\n" +output_of_this_epoch

        # The rest is based on epoch dev evaluation.
        if conf["keep_best_model"] == "True":
            if epoch_dev>max_eval_val:
                ddir=conf['model_save_path'] + model_conf 
                shutil.rmtree(ddir, ignore_errors=True) 
                os.makedirs(ddir,exist_ok=True)
                max_eval_val=epoch_dev
                save_model_to_location(save_location, model, output_of_this_epoch)
        else:
            save_model_to_location(save_location, model, output_of_this_epoch)

        if epoch>conf["min_num_epochs"] and conf["check_last_num_models_to_stop"]>0 and epoch>conf["check_last_num_models_to_stop"]:
            maxin_last = 0
            for i in range(len(epoch_devs)-conf["check_last_num_models_to_stop"], len(epoch_devs)):
                if epoch_devs[i]>maxin_last:
                    maxin_last = epoch_devs[i]
            maxin_rest = 0
            for i in range(0,len(epoch_devs)-conf["check_last_num_models_to_stop"]):
                if epoch_devs[i]>maxin_rest:
                    maxin_rest = epoch_devs[i]

            if maxin_last <= maxin_rest:
                print("For the last "+str(conf["check_last_num_models_to_stop"]) + " epochs no improvement happened. Stopping training for "+model_conf)
                epoch_devs[epoch]=epoch_dev
                print("Epoch dev evaluations:")
                print(epoch_devs)
                print("Epoch test evaluations:")
                print(epoch_tests)
                print("Epoch eval evaluations:")
                print(epoch_evals)
                break
    f=open(conf['model_save_path'] + model_conf +"/history","w")
    f.write(all_output)
    f.close()
    f=open(conf['model_save_path'] + model_conf +"/test_dev_eval.json","w")
    f.write(json.dumps({'epoch_losses':epoch_losses, 'epoch_devs':epoch_devs,'epoch_tests':epoch_tests,'epoch_evals': epoch_evals}))
    f.close()
