from transformers import GPT2Config, GPT2Tokenizer
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

tokenizer = GPT2Tokenizer.from_pretrained(save_path)

tokenizer.add_special_tokens({
    "eos_token":"</s>",
    "bos_token":"<s>",
    "unk_token":"<unk>",
    "pad_token":"<pad>",
    "mask_token":"<mask>"
})

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_ctx= 1024
)

model = GPT2LMHeadModel(config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/nepali-wikipedia/output.txt",
    block_size=128,
)

train_dataset,test_dataset = train_test_split(dataset,test_size=0.2)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)

train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=data_collator)
eval_dataloader = DataLoader(test_dataset,collate_fn=data_collator,batch_size=32)

optimizer = AdamW(model.parameters(),lr=2e-5)
device =  'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
  #training
  model.train()
  total_loss = 0
  for step, batch in enumerate(train_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    logits, _ = model(batch['input_ids'])  # Assuming the model returns logits and presents

    # Calculate the loss manually
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
    loss = loss_fct(logits.view(-1, logits.size(-1)), batch['input_ids'].view(-1))

    # Clear the previous gradients
    optimizer.zero_grad()

    # Compute gradients
    loss.backward()

    # Update the model's parameters
    optimizer.step()

    total_loss += loss.item()
    progress_bar.update(1)

  average_loss = total_loss / len(train_dataloader)
  print(f"Epoch: {epoch+1} Average Loss: {average_loss:.4f}")

torch.save(model.state_dict(), '/nep-gpt/pytorch_model.bin')
config.save_pretrained('/nep-gpt')
