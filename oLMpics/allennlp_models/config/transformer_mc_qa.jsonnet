local train_size = 200;
local batch_size = 8;
local gradient_accumulation_batch_size = 2;
local num_epochs = 1;
local learning_rate = 0;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "roberta-large";
//local transformer_model = "bert-large-uncased-whole-word-masking";
//local transformer_model = "bert-large-uncased";
//local transformer_model = "bert-base-uncased";
local cuda_device = 0;

{
  "dataset_reader": {
    "type": "transformer_mc_qa",
    "sample": 1,
    "num_choices": 3,
    //"add_prefix": {"q": "Q: ", "a": "A: "},
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  "validation_dataset_reader": {
    "type": "transformer_mc_qa",
    "sample": -1,
    "num_choices": 3,
    //"add_prefix": {"q": "Q: ", "a": "A: "},
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  //"datasets_for_vocab_creation": [],
  "train_data_path": "https://olmpics.s3.us-east-2.amazonaws.com/challenge/conjunction/conjunction_filt4_train.jsonl.gz",
  "validation_data_path": "https://olmpics.s3.us-east-2.amazonaws.com/challenge/conjunction/conjunction_filt4_dev.jsonl.gz",

  "model": {
    "requires_grad":false,
    "type": "transformer_mc_qa",
    "pretrained_model": transformer_model
  },
  "data_loader": {
    "batch_size": batch_size,
    "shuffle": true
  },
  "trainer": {
    "callbacks": [
      {
        type: "log_metrics_to_wandb",
      }
    ],
    "optimizer": {
      "type": "adamw",
      "weight_decay": weight_decay,
      "betas": [0.9, 0.98],
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": learning_rate
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": warmup_ratio,
      "num_steps_per_epoch": std.ceil(train_size / batch_size /  gradient_accumulation_batch_size),
    },
    "validation_metric": "+EM",
    // "num_serialized_models_to_keep": 1,
    // "should_log_learning_rate": true,
    "num_gradient_accumulation_steps": gradient_accumulation_batch_size,
    // "grad_clipping": 1.0,
    "num_epochs": num_epochs,
    "cuda_device": cuda_device,
    "run_sanity_checks": false
  }
}