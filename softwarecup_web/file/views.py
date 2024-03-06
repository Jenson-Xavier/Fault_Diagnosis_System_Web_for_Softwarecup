from pyecharts.globals import ThemeType
from pyecharts.faker import Faker
from pyecharts.charts import Bar, Pie, Page
from pyecharts.charts import Bar
from pyecharts import options as opts
from django.utils.http import *
from django.utils import *
import os
from django.urls import *
from django.http import *
from .models import *
from .forms import *
from django.shortcuts import render

from ML.project_model import *
from ML.train import *
from ML.Inference import *

from jinja2 import Environment, FileSystemLoader
from pyecharts.globals import CurrentConfig
CurrentConfig.GLOBAL_ENV = Environment(
    loader=FileSystemLoader("/myweb/softwarecup_web/file/templates"))

# Create your views here.

# file应用的视图

# html渲染主视图


def index_view(request):
    return render(request, "index.html")

    # c = (
    #     Bar()
    #     .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    #     .add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
    #     .add_yaxis("商家B", [15, 25, 16, 55, 48, 8])
    #     .set_global_opts(title_opts=opts.TitleOpts(title="Bar-基本示例", subtitle="我是副标题"))
    # )
    # html = c.render_embed()
    # return HttpResponse(html)


def train_view(request):
    # 渲染上传训练数据文件csv文件视图
    if request.method == 'POST':
        train_form = FileForm_train(request.POST, request.FILES)
        train_file_model = None
        if train_form.is_valid():
            train_files = request.FILES.getlist('file')
            for train_file in train_files:
                file_path = os.path.join(
                    './upload_train', train_file.name)
                file_model = FileModel(name=train_file.name, path=file_path)
                train_file_model = file_model
                file_model.save()
                file_des = open(file_path, 'wb+')
                for chunk in train_file.chunks():
                    file_des.write(chunk)
                file_des.close()
                # 虽然采用列表循环的形式
                # 事实上只允许上传一个训练数据集文件
                gb_train_file_path = file_path
        return HttpResponseRedirect(reverse('training', args=(train_file_model.pk,)))
    else:
        train_form = FileForm_train()
        return render(request, 'upload_train.html', locals())


def training_view(request, id):
    file_result = FileModel.objects.filter(id=id)
    if file_result:
        train_file = list(file_result)[0]
        context = {
            'train_file_name': train_file.name,
            'train_id': id
        }
        return render(request, "training.html", context)
    else:
        return HttpResponse("训练文件上传失败！！")


def train_model_download_view(request, id):
    file_result = FileModel.objects.filter(id=id)
    if file_result:
        train_file = list(file_result)[0]
        train_path = train_file.path
        file = open(train_path, 'rb')
        train(file)
        return render(request, "train_downloading.html")
    else:
        return HttpResponse("找不到正确的训练数据文件，无法进行训练！！")


def train_downloading_view(request):
    return FileResponse(open("/myweb/softwarecup_web/ML_models/target_model.pt", 'rb'),
                        as_attachment=True)


def test_view(request):
    if request.method == 'POST':
        pred_form = FileForm_test(request.POST, request.FILES)
        pred_file_model = None
        if pred_form.is_valid():
            pred_files = request.FILES.getlist('file')
            for pred_file in pred_files:
                file_path = os.path.join(
                    './upload_test', pred_file.name)
                file_model = FileModel(name=pred_file.name, path=file_path)
                pred_file_model = file_model
                file_model.save()
                file_des = open(file_path, 'wb+')
                for chunk in pred_file.chunks():
                    file_des.write(chunk)
                file_des.close()
                # 虽然采用列表循环的形式
                # 事实上只允许上传一个训练数据集文件
                gb_train_file_path = file_path
        return HttpResponseRedirect(reverse('testing', args=(pred_file_model.pk,)))
    else:
        test_form = FileForm_test()
        return render(request, 'upload_test.html', locals())


def testing_view(request, id):
    file_result = FileModel.objects.filter(id=id)
    if file_result:
        test_file = list(file_result)[0]
        context = {
            'test_file_name': test_file.name,
            'test_id': id
        }
        return render(request, "testing.html", context)
    else:
        return HttpResponse("测试文件上传失败！！")


# 柱形图1
def bar_1(pred_counts) -> Bar:
    xaxis = ["故障0", "故障1", "故障2", "故障3", "故障4", "故障5"]
    c = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
        .add_xaxis(xaxis)
        .add_yaxis("数量", pred_counts, itemstyle_opts=opts.ItemStyleOpts(color="#00CD96"))
        # .add_yaxis("比例", pred_proportion, itemstyle_opts=opts.ItemStyleOpts(color="#00CD00"))
        .set_global_opts(
            title_opts={"text": "预测故障类别的数量"},
            brush_opts=opts.BrushOpts(),            # 设置操作图表的画笔功能
            toolbox_opts=opts.ToolboxOpts(),        # 设置操作图表的工具箱功能
            yaxis_opts=opts.AxisOpts(name="数量"),
            xaxis_opts=opts.AxisOpts(name="故障类别"),
        )
    )
    return c


# 饼图
def pie_rosetype(pred_proportion) -> Pie:
    v = ["故障0", "故障1", "故障2",
         "故障3", "故障4", "故障5"]
    c = (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
        .add(
            "",
            [list(z) for z in zip(v, pred_proportion)],
            radius=["30%", "75%"],
            center=["50%", "50%"],
            rosetype="radius",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="预测故障类别的比例"))
    )
    return c


# 柱状图2
def bar_2(precision_res, recall_res, macro_f1_res) -> Bar:
    xaxis = ["故障0", "故障1", "故障2", "故障3", "故障4", "故障5"]
    c = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
        .add_xaxis(xaxis)
        .add_yaxis("精确率", precision_res)
        .add_yaxis("召回率", recall_res)
        .add_yaxis("macro_f1", macro_f1_res)
        .set_global_opts(
            title_opts={"text": "各故障类别的预测效果"},
            brush_opts=opts.BrushOpts(),            # 设置操作图表的画笔功能
            toolbox_opts=opts.ToolboxOpts(),        # 设置操作图表的工具箱功能
            yaxis_opts=opts.AxisOpts(name="评估指标"),
            xaxis_opts=opts.AxisOpts(name="故障类别"),
        )
    )
    return c


def page_simple_layout_with_label(pred_counts, pred_proportion, precision_res, recall_res, macro_f1_res):
    page = Page(layout=Page.SimplePageLayout)
    page.add(
        bar_1(pred_counts),
        pie_rosetype(pred_proportion),
        bar_2(precision_res, recall_res, macro_f1_res)
    )
    return page.render_embed()


def page_simple_layout_without_label(pred_counts, pred_proportion):
    page = Page(layout=Page.SimplePageLayout)
    page.add(
        bar_1(pred_counts),
        pie_rosetype(pred_proportion),
    )
    return page.render_embed()


def test_download_view(request, id):
    file_result = FileModel.objects.filter(id=id)
    if file_result:
        test_file = list(file_result)[0]
        test_path = test_file.path
        file = open(test_path, 'rb')
        json_text, pred_counts, pred_proportion, labels, macro_f1_score, precision_res, recall_res, macro_f1_res = inference(
            file)
        context = {
            'dict': json_text,
            'score': macro_f1_score,
            'flag': labels
        }

        html = None
        if labels is not None:
            html = page_simple_layout_with_label(
                pred_counts, pred_proportion, precision_res, recall_res, macro_f1_res)
        else:
            html = page_simple_layout_without_label(
                pred_counts, pred_proportion)
        with open('/myweb/softwarecup_web/file/templates/visualization.html', 'w') as f:
            f.write(html)

        return render(request, "test_show.html", context)
    else:
        return HttpResponse("找不到正确的测试数据文件，无法进行训练！！")


def test_downloading_view(request):
    return FileResponse(open("/myweb/softwarecup_web/ML_pred_results/submit.json", 'rb'), as_attachment=True)


def test_visualization_view(request):
    return render(request, "visualization.html")
