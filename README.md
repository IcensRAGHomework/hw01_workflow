# Prompt Engineer 作業題目

## 作業內容

請使用 **LangChain** 套件完成以下作業，並實作提供的方法 `generate(question, use_tools_call)`。實作於新建的檔案 **`student_assignment.py`** 中。

---

### 作業1

1. **問題**：`2024年台灣10月紀念日有哪些?`(範例)
2. **方法**：實作 `generate(question, use_tools_call=False)`，用於回答上述問題。
3. **輸出格式**：
   - JSON 格式如下：
     ```json
     {
         "Result": [
             {
                 "日期": "2024-10-10",
                 "名稱": "國慶日"
             }
         ]
     }
     ```

---

### 作業2

1. **問題**：`2024年台灣10月紀念日有哪些?`(範例)
2. **方法**：
   - 使用 Function Calling 的方式查詢指定的 API。
   - 實作 `generate(question, use_tools_call=True)`，用於回答上述問題。
3. **指定 API**：
   - 使用 [Calendarific API](https://calendarific.com/)。
   - 步驟：
     1. 訪問 Calendarific 網站並註冊帳戶。
     2. 登錄後進入 Dashboard，取得您的 API Key。
4. **輸出格式**：
   - JSON 格式如下：
     ```json
     {
         "Result": [
             {
                 "日期": "2024-10-10",
                 "名稱": "國慶日"
             }
         ]
     }
     ```

### 作業3

1. **問題**：`根據先前的節日清單，是否應該新增以下這個節日？{"date": "10-31", "name": "蔣公誕辰紀念日"}'`(範例)
2. **方法**：
   - 使用 ConversationBufferMemory 的方式記憶前一次的回答。
   - 實作 `generate(question, use_tools_call=True)`，用於回答上述問題。
3. **輸出格式**：
   - JSON 格式如下：
     ```json
     {
         "Result": [
             {
                 "add": false,
                 "reason": "蔣中正的生日並未被廣泛視為台灣的公共假日，將其列入可能與當前的文化或政治情感不符。"
             }
         ]
     }
     ```

---



### 注意事項
- 必須使用 **LangChain** 套件完成方法實作。
- 確保輸出的格式與範例一致。
