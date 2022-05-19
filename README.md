# LabelApp
## deepedit
```shell
monailabel start_server --app radiology --studies my_dataset --conf models deepedit
```
## cardiac segmentation
```shell
monailabel start_server --app radiology --studies mydata --conf models segmentation_cardiac --conf skip_scoring false --conf skip_strategies false --conf tta_enabled true
```