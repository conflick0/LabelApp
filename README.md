# LabelApp
## deepedit
```shell
monailabel start_server --app radiology --studies mydata --conf models deepedit
```
## cardiac segmentation
* unetr
```shell
monailabel start_server --app radiology --studies mydata --conf models segmentation_cardiac --conf skip_scoring false --conf skip_strategies false --conf tta_enabled true
```
* swin unetr
```shell
monailabel start_server --app radiology --studies mydata --conf models seg_cardiac_swin_unetr --conf skip_scoring false --conf skip_strategies false --conf tta_enabled true
```