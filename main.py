import tkinter as TK
import customtkinter as CTK
import analyze
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# define initial window themes
CTK.set_appearance_mode("System")
CTK.set_default_color_theme("green")

# ============== DEFINE GUI WINDOW ============= #


# Defining main window
class App(CTK.CTk):
    """
    defines the main GUI window
    """
    WIDTH = 780
    HEIGHT = 520

    def __init__(self):
        super().__init__()

        self.title("News Analysis Tool")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============ #

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = CTK.CTkFrame(master=self, width=180)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = CTK.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ Left Frame =================== #

        self.frame_left.grid_rowconfigure(0, minsize=20)
        self.frame_left.grid_rowconfigure(5, weight=1)
        self.frame_left.grid_rowconfigure(8, minsize=20)
        self.frame_left.grid_rowconfigure(11, minsize=10)

        self.window_label = CTK.CTkLabel(master=self.frame_left,
                                         text="Article Analysis Tool",
                                         font=("Lucida Sans", -16))
        self.window_label.grid(row=1, column=0, pady=10, padx=10)

        self.style_label = CTK.CTkLabel(master=self.frame_left, text="Style Options:", font=("Lucida Sans", -12))
        self.style_label.grid(row=9, column=0, pady=0, padx=10)

        self.style_optionmenu = CTK.CTkOptionMenu(self.frame_left,
                                                  values=["Light", "Dark", "System"],
                                                  command=self.change_appearance_mode)
        self.style_optionmenu.grid(row=10, column=0, pady=5, padx=10)
        self.style_optionmenu.set('Select Style')

        # ============ Right Frame ================== #

        self.frame_right.rowconfigure((0, 1), weight=0)
        self.frame_right.rowconfigure(2, weight=10)
        self.frame_right.columnconfigure(0, weight=1)

        # Text entry for Title
        self.text_entry = CTK.CTkEntry(master=self.frame_right,
                                       width=256,
                                       placeholder_text="Enter Article Title Here...")
        self.text_entry.grid(row=0, column=0, pady=10, padx=10, sticky="we")

        self.text_frame = CTK.CTkFrame(master=self.frame_right)
        self.text_frame.rowconfigure(0, weight=1)
        self.text_frame.rowconfigure(1, weight=15)
        self.text_frame.rowconfigure(2, weight=1)
        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.grid(row=2, column=0, pady=10, padx=10, sticky="nswe")

        # Calls Analysis functions and opens new window.
        self.input_button = CTK.CTkButton(master=self.text_frame, text="Analyze Article", command=self.load_text)
        self.input_button.grid(row=2, column=0, pady=10, padx=10, sticky="we")

        self.article_text_label = CTK.CTkLabel(master=self.text_frame,
                                               text='Article Text:',
                                               font=("Lucida Sans", -12))
        self.article_text_label.grid(row=0, column=0, pady=5, padx=5, sticky='w')

        # Contains Article text
        self.text_display = CTK.CTkTextbox(master=self.text_frame, corner_radius=5, font=("Lucida Sans", -16))
        self.text_display.grid(row=1, column=0, pady=5, padx=5, sticky="nswe")

    # Collects title and text into variables and creates an analysis window
    def load_text(self):
        title = self.text_entry.get()
        text = self.text_display.get("0.0", "end")

        if len(text) >= 50:

            self.analyze_text(title, text)
            return

        # If the text isn't large enough to analyze, create error dialogue
        # ============= create error window ========== #
        error_window = CTK.CTkToplevel(self)
        error_window.title("ERROR")
        error_window.geometry("360x128")

        error_frame = CTK.CTkFrame(master=error_window)
        error_frame.rowconfigure(0, weight=20)
        error_frame.rowconfigure(1, weight=1)
        error_frame.grid(pady=10, padx=10, sticky="nswe")

        error_label = CTK.CTkLabel(master=error_frame,
                                   text="ERROR! NOT ENOUGH TEXT TO ANALYZE",
                                   font=("Lucida Sans", -16))
        error_label.grid(row=0, column=0, pady=10, padx=10, sticky="nswe")

        # Confirmation button closes error window
        error_button = CTK.CTkButton(master=error_frame, text="OK", command=lambda: error_window.destroy())
        error_button.grid(row=1, column=0, pady=10, padx=10, sticky="nswe")

        return

    def clear_textboxes(self):
        """
        Clears text_entry and text_display boxes of all characters.
        return None
        """
        self.text_entry.delete(0, "end")
        self.text_display.delete('1.0', "end")
        return

    def analyze_text(self, title, text):
        """
        Calls Analyze function from analyze.py and then builds the analysis window containing three graphs to illustrate
         the text data input
        :param title: The article title to be analyzed
        :param text: the article text to be analyzed
        :return: None
        """
        ##################################################
        # ====== Analysis function creates graphs ====== #

        cleaned_text, tf_graph, pie_chart, prediction = analyze.analyze(title, text)

        # ========== create analysis window ============ #
        analysis_window = CTK.CTkToplevel(self)
        analysis_window.geometry("1600x800")
        analysis_window.rowconfigure(0, weight=3)
        analysis_window.rowconfigure(1, weight=1)
        analysis_window.columnconfigure(0, weight=3)
        analysis_window.columnconfigure(1, weight=1)

        ##################################################
        # =========== Create graph display ============= #
        graph_frame = CTK.CTkFrame(master=analysis_window, corner_radius=5, border_width=5)
        graph_frame.rowconfigure(0, weight=1)
        graph_frame.rowconfigure(1, weight=5)
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.columnconfigure((1, 2, 3), weight=7)
        graph_frame.columnconfigure(4, weight=1)
        graph_frame.grid(row=0, column=0, pady=10, padx=10, sticky="nswe", columnspan=2)

        # == List of most common words in a wordcloud ========= #
        graph_1_frame = CTK.CTkFrame(master=graph_frame, corner_radius=5, border_width=2)
        graph_1_frame.grid(row=1, column=1, pady=5, padx=3, ipadx=10, ipady=10, sticky="nswe")

        wordcloud_label = CTK.CTkLabel(text="Word Cloud of 25 important words",
                                       font=("Lucida Sans", -20),
                                       master=graph_1_frame)
        wordcloud_label.pack()

        width = 400
        height = 500

        wordcloud = analyze.build_wordcloud(cleaned_text, width, height)

        f = Figure(figsize=(4, 5), dpi=100)
        f.figimage(wordcloud)

        canvas_1 = FigureCanvasTkAgg(f, master=graph_1_frame)
        canvas_1.draw()
        canvas_1.get_tk_widget().pack()
        toolbar_1 = NavigationToolbar2Tk(canvas_1, graph_1_frame)
        toolbar_1.update()
        canvas_1.get_tk_widget().pack()

        # ===== 3 Bar Barchart to display word frequency ======= #
        graph_2_frame = CTK.CTkFrame(master=graph_frame, corner_radius=5, border_width=2)
        graph_2_frame.grid(row=1, column=2, pady=5, padx=3, ipadx=10, ipady=10, sticky="nswe")

        f_1 = Figure(figsize=(4, 5), dpi=100)
        a = f_1.add_subplot(111)

        a.bar(x=tf_graph[0],
              height=tf_graph[1],
              tick_label=tf_graph[0])

        a.set_ylabel("Frequency")
        a.yaxis.get_major_locator().set_params(integer=True)

        title_2_label = CTK.CTkLabel(text="Top 3 Most Common Words",
                                     font=("Lucida Sans", -20),
                                     master=graph_2_frame)
        title_2_label.pack()

        canvas_2 = FigureCanvasTkAgg(f_1, master=graph_2_frame)
        canvas_2.draw()
        canvas_2.get_tk_widget().pack()
        toolbar_2 = NavigationToolbar2Tk(canvas_2, graph_2_frame)
        toolbar_2.update()
        canvas_2.get_tk_widget().pack()

        # == Pie Chart to display level of confidence ======== #
        graph_3_frame = CTK.CTkFrame(master=graph_frame, corner_radius=5, border_width=2)
        graph_3_frame.grid(row=1, column=3, pady=5, padx=3, ipadx=10, ipady=10, sticky="nswe")

        f_3 = Figure(figsize=(4, 5), dpi=100)
        b = f_3.add_subplot(111)

        b.pie(pie_chart,
              colors=['firebrick', 'forestgreen'],
              startangle=90,
              shadow=True,
              labels=["Fake", "True"],
              autopct='%1.1f%%')

        pie_title = CTK.CTkLabel(text="How confident is this prediction?",
                                 font=("Lucida Sans", -20),
                                 master=graph_3_frame)
        pie_title.pack()

        canvas_3 = FigureCanvasTkAgg(f_3, master=graph_3_frame)
        canvas_3.draw()
        canvas_3.get_tk_widget().pack()
        toolbar_3 = NavigationToolbar2Tk(canvas_3, graph_3_frame)
        toolbar_3.update()
        canvas_3.get_tk_widget().pack()

        ##################################################
        # ======== Create analysis results frame ======= #
        options_frame = CTK.CTkFrame(master=analysis_window, corner_radius=5)
        options_frame.grid(row=1, column=0, pady=10, padx=10, sticky="nswe")
        options_frame.rowconfigure((0, 1), weight=1)
        options_frame.columnconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), weight=1)

        prediction_text = 'FAKE'
        if prediction == 1:
            prediction_text = 'TRUE'

        result_label = CTK.CTkLabel(master=options_frame,
                                    text=f'The article is MOST LIKELY: \n'
                                         f'|   {prediction_text}   |',
                                    font=("Lucida Sans", -20),
                                    anchor=TK.CENTER)
        result_label.grid(row=0, column=2, columnspan=5, padx=10, pady=10)

        ##################################################
        # ====== Create right hand options panel ======= #

        result_frame = CTK.CTkFrame(master=analysis_window, corner_radius=5)
        result_frame.rowconfigure((0, 1), weight=1)
        result_frame.columnconfigure((0, 1), weight=1)
        result_frame.grid(row=1, column=1, pady=10, padx=10, rowspan=2, sticky="nswe")

        options_label = CTK.CTkLabel(text='analyze another article?',
                                     font=("Lucida Sans", -20),
                                     master=result_frame)
        options_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2, sticky='')

        try_another_button = CTK.CTkButton(text="try another",
                                           master=result_frame,
                                           command=lambda: [analysis_window.destroy(), self.clear_textboxes()])
        try_another_button.grid(row=1, column=0, padx=10, pady=10)

        close_button = CTK.CTkButton(text="close",
                                     master=result_frame,
                                     command=lambda: [analysis_window.destroy(), self.quit(), self.destroy()])
        close_button.grid(row=1, column=1, padx=10, pady=10)

    def change_appearance_mode(self, new_appearance_mode):
        CTK.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.quit()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
