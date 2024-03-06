from pyecharts.charts import Bar, Pie, Page
from pyecharts.faker import Faker
from pyecharts.globals import ThemeType
from pyecharts import options as opts

from ML.Inference import *

json_text, pred_counts, pred_proportion, labels,\
    macro_f1_score, precision_res, recall_res, macro_f1_res = inference(
        '/myweb/softwarecup_web/ML/validate_1000.csv')


# 柱形图1
def bar_1() -> Bar:
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
def pie_rosetype() -> Pie:
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
def bar_2() -> Bar:
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


def page_simple_layout_with_label():
    page = Page(layout=Page.SimplePageLayout)
    page.add(
        bar_1(),
        pie_rosetype(),
        bar_2()
    )
    page.render("分类结果查看.html")


def page_simple_layout_without_label():
    page = Page(layout=Page.SimplePageLayout)
    page.add(
        bar_1(),
        pie_rosetype(),
    )
    page.render("分类结果查看.html")


if __name__ == "__main__":
    if labels is not None:
        page_simple_layout_with_label()
    else:
        page_simple_layout_without_label()
