---
category: Jekyll
layout: post
tags:
  - jekyll
title: '写 Jekyll 博客的正确姿势'
---


用 Jekyll 写博客有个麻烦的地方，就是你可能需要把博客`git pull`下来，修改完成`commit`后，再`push`上 GitHub；或者使用 GitHub 的在线编辑器编辑，无论哪种都非常麻烦。幸好找到了几个第三方的写博客和图床工具。

<!--more-->

## 写博客工具

有两个工具，分别是`Jekyll Editor`、`prose.io`。

### Jekyll Editor

![Jekyll Editor](http://simpleyyt.qiniudn.com/15-10-11/10214115.jpg)

Jekyll Editor 专门为 Jekyll 博客定制的强大的 markdown 编辑器，它会自动从`<yourname>.github.io`仓库读取`_post`目录下的博客列表，并可以读取、创建、修改博客。

 * **Chrome 商店**：https://chrome.google.com/webstore/detail/jekyll-editor/dfdkgbhjmllemfblfoohhehdigokocme
 
### Prose.io

非常好的一个工具，它的编辑器非常强大，可以上传图片、文件等，可以在`_config.yml`文件中配置`prose.io`。

![Prose.io](http://simpleyyt.qiniudn.com/15-9-21/82332870.jpg)

缺点就是不支持实时预览，而且也不会自己保存。

## 图床

图床的话强烈推荐**七牛**，其缺点就是操作不人性化，但是 chrome 上面有好多相关的插件解决这个问题，比如[极简图床](http://yotuku.cn/) 便是基于七牛的图床网站。

## 图表工具

写博客难免会需要用画一些图表，有两类图表，一类是 [yUML](http://yuml.me/diagram/scruffy/class/draw)、[plantUML](http://plantuml.com/), 另一类是 [draw.io](http://draw.io)。

### yUML 和 plantUML

这类图表只需要按格式输入代码，便会自动产生图表，生成图片链接，省去了上传到图床，后期也可以修改。

![yUML](http://simpleyyt.qiniudn.com/15-9-21/46889912.jpg)

![plantUML](http://simpleyyt.qiniudn.com/15-9-21/34152859.jpg)

### draw.io

这个是在线手动绘图的工具，chrome 应用商店里面也下载得到离线应用，绘图完成之后需要上传到图床中。

![draw.io](http://simpleyyt.qiniudn.com/15-9-21/68984484.jpg)
