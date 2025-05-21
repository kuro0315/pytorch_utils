# pytorch_utils
pytorch使っているときの便利関数

## 追加されたユーティリティ

- `dropout_utils.dropout_train_only(model)` : モデル中の `nn.Dropout` 系レイヤだけを
  `train` モードにし、それ以外を `eval` モードに設定します。子モジュールも再帰的に
  走査するため、ネストされたドロップアウト層も確実に `train` になります。
