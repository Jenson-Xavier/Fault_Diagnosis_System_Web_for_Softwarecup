# 文件上传的表单 通过视图来渲染

from django import forms


class FileForm_train(forms.Form):
    file = forms.FileField(widget=forms.ClearableFileInput(
        attrs={'multiple': True}), label='请选择你要上传的训练数据文件(仅支持csv文件并且内容格式需要与大赛提供的一致)')


class FileForm_test(forms.Form):
    file = forms.FileField(widget=forms.ClearableFileInput(
        attrs={'multiple': True}), label='请选择你要上传的测试数据文件(仅支持csv文件并且内容格式需要与大赛提供的一致,允许带标签列)')
