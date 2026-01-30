import cv2
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class MaskVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask标签可视化工具")
        self.root.geometry("1400x800")
        
        # 图像相关变量
        self.folder_path = None
        self.image_pairs = []  # [(原图路径, mask路径), ...]
        self.current_index = 0
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 顶部控制面板
        control_frame = tk.Frame(self.root, height=60, bg='lightgray')
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 选择文件夹按钮
        self.btn_select = tk.Button(control_frame, text="选择文件夹", 
                                    command=self.select_folder, 
                                    font=('Arial', 12), 
                                    bg='#4CAF50', fg='white', 
                                    padx=20, pady=10)
        self.btn_select.pack(side=tk.LEFT, padx=5)
        
        # 文件夹路径显示
        self.lbl_path = tk.Label(control_frame, text="未选择文件夹", 
                                 font=('Arial', 10), bg='lightgray')
        self.lbl_path.pack(side=tk.LEFT, padx=10)
        
        # 图像计数显示
        self.lbl_count = tk.Label(control_frame, text="0/0", 
                                  font=('Arial', 12, 'bold'), bg='lightgray')
        self.lbl_count.pack(side=tk.RIGHT, padx=10)
        
        # 导航按钮框架
        nav_frame = tk.Frame(self.root, height=60, bg='lightgray')
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # 上一张按钮
        self.btn_prev = tk.Button(nav_frame, text="◀ 上一张 (A)", 
                                  command=self.prev_image, 
                                  font=('Arial', 12), 
                                  state=tk.DISABLED,
                                  padx=20, pady=10)
        self.btn_prev.pack(side=tk.LEFT, padx=20)
        
        # 下一张按钮
        self.btn_next = tk.Button(nav_frame, text="下一张 (D) ▶", 
                                  command=self.next_image, 
                                  font=('Arial', 12), 
                                  state=tk.DISABLED,
                                  padx=20, pady=10)
        self.btn_next.pack(side=tk.RIGHT, padx=20)
        
        # 当前文件名显示
        self.lbl_filename = tk.Label(nav_frame, text="", 
                                     font=('Arial', 11), bg='lightgray')
        self.lbl_filename.pack(side=tk.TOP, pady=5)
        
        # 图像显示区域
        display_frame = tk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：原图
        left_frame = tk.Frame(display_frame, borderwidth=2, relief=tk.GROOVE)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        lbl_original_title = tk.Label(left_frame, text="原图", 
                                      font=('Arial', 14, 'bold'), 
                                      bg='#2196F3', fg='white')
        lbl_original_title.pack(side=tk.TOP, fill=tk.X)
        
        self.canvas_original = tk.Canvas(left_frame, bg='white')
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        
        # 中间：Mask标签
        middle_frame = tk.Frame(display_frame, borderwidth=2, relief=tk.GROOVE)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        lbl_mask_title = tk.Label(middle_frame, text="Mask标签", 
                                  font=('Arial', 14, 'bold'), 
                                  bg='#FF9800', fg='white')
        lbl_mask_title.pack(side=tk.TOP, fill=tk.X)
        
        self.canvas_mask = tk.Canvas(middle_frame, bg='white')
        self.canvas_mask.pack(fill=tk.BOTH, expand=True)
        
        # 右侧：叠加显示
        right_frame = tk.Frame(display_frame, borderwidth=2, relief=tk.GROOVE)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        lbl_overlay_title = tk.Label(right_frame, text="叠加显示", 
                                     font=('Arial', 14, 'bold'), 
                                     bg='#F44336', fg='white')
        lbl_overlay_title.pack(side=tk.TOP, fill=tk.X)
        
        self.canvas_overlay = tk.Canvas(right_frame, bg='white')
        self.canvas_overlay.pack(fill=tk.BOTH, expand=True)
        
        # 绑定键盘事件
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('a', lambda e: self.prev_image())
        self.root.bind('d', lambda e: self.next_image())
        self.root.bind('A', lambda e: self.prev_image())
        self.root.bind('D', lambda e: self.next_image())
        
    def select_folder(self):
        """选择包含图像的文件夹"""
        folder = filedialog.askdirectory(title="选择包含图像的文件夹")
        if not folder:
            return
        
        self.folder_path = folder
        self.lbl_path.config(text=f"当前文件夹: {folder}")
        
        # 查找图像对
        self.load_image_pairs()
        
    def load_image_pairs(self):
        """加载文件夹中的图像对"""
        self.image_pairs = []
        
        if not self.folder_path:
            return
        
        # 获取文件夹中的所有.bmp文件
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.bmp')]
        
        # 找到所有原图（不以_t.bmp结尾的）
        original_images = [f for f in files if not f.endswith('_t.bmp')]
        
        # 为每个原图查找对应的mask
        for orig in original_images:
            mask_name = orig.replace('.bmp', '_t.bmp')
            mask_path = os.path.join(self.folder_path, mask_name)
            
            if os.path.exists(mask_path):
                orig_path = os.path.join(self.folder_path, orig)
                self.image_pairs.append((orig_path, mask_path))
        
        if len(self.image_pairs) == 0:
            messagebox.showwarning("警告", "未找到有效的图像对！\n请确保mask文件命名为'原图名称_t.bmp'")
            return
        
        # 重置索引
        self.current_index = 0
        
        # 启用导航按钮
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.NORMAL)
        
        # 显示第一张图像
        self.display_current_image()
        
    def display_current_image(self):
        """显示当前索引的图像"""
        if not self.image_pairs:
            return
        
        orig_path, mask_path = self.image_pairs[self.current_index]
        
        # 更新计数和文件名
        self.lbl_count.config(text=f"{self.current_index + 1}/{len(self.image_pairs)}")
        self.lbl_filename.config(text=os.path.basename(orig_path))
        
        # 读取图像
        img_orig = cv2.imread(orig_path)
        img_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img_orig is None or img_mask_raw is None:
            messagebox.showerror("错误", f"无法读取图像:\n{orig_path}\n或\n{mask_path}")
            return
        
        # 将BGR转为RGB
        img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        
        # 增强mask的可见度
        # 检查mask的值范围，如果太小则放大
        # 使用cv2操作避免numpy问题
        mask_scaled = cv2.multiply(img_mask_raw, 255)  # 先尝试放大255倍
        
        # 如果mask值很小，进一步处理
        # 创建一个二值化版本用于更好的可视化
        _, mask_binary = cv2.threshold(img_mask_raw, 0, 255, cv2.THRESH_BINARY)
        
        # 使用二值化版本作为可视化的mask
        mask_vis = mask_binary
        
        # 将mask转为伪彩色以便更好地可视化
        mask_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        mask_colored_rgb = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
        
        # 创建叠加图像 - 使用红色高亮
        # 创建一个红色mask图层
        height, width = img_mask_raw.shape
        mask_red = cv2.merge([
            cv2.multiply(mask_binary, 0),      # B通道 = 0
            cv2.multiply(mask_binary, 0),      # G通道 = 0  
            mask_binary                        # R通道 = mask
        ])
        
        # 混合图像 (都是RGB格式)
        overlay = cv2.addWeighted(img_orig_rgb, 0.7, mask_red, 0.3, 0)
        
        # 显示图像
        self.show_image_on_canvas(self.canvas_original, img_orig_rgb)
        self.show_image_on_canvas(self.canvas_mask, mask_colored_rgb)
        self.show_image_on_canvas(self.canvas_overlay, overlay)
        
    def show_image_on_canvas(self, canvas, img_array):
        """在canvas上显示图像"""
        # 获取canvas尺寸
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 600
        
        # 将numpy数组转为PIL Image
        img_pil = Image.fromarray(img_array)
        
        # 计算缩放比例以适应canvas
        img_width, img_height = img_pil.size
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h) * 0.95  # 留一点边距
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # 缩放图像
        img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转为PhotoImage
        img_tk = ImageTk.PhotoImage(img_resized)
        
        # 清除canvas并显示图像
        canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
        
        # 保持引用，防止被垃圾回收
        canvas.image = img_tk
        
    def prev_image(self):
        """显示上一张图像"""
        if not self.image_pairs:
            return
        
        self.current_index = (self.current_index - 1) % len(self.image_pairs)
        self.display_current_image()
        
    def next_image(self):
        """显示下一张图像"""
        if not self.image_pairs:
            return
        
        self.current_index = (self.current_index + 1) % len(self.image_pairs)
        self.display_current_image()


def main():
    root = tk.Tk()
    app = MaskVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
