# Writing riddles for your community

This guide is for native speakers who want to add their community to Cultural Riddles. It explains what we look for in a riddle, how to choose your topics, and how to fill in the workbook. When your riddles are ready, [CONTRIBUTING.md](../CONTRIBUTING.md) walks you through checking and submitting them.

## What we are looking for

A cultural riddle describes something without naming it. Someone who grew up in your community should recognise the answer at once. Someone who only knows the language, or who searches the web, should struggle.

Good riddles draw on lived experience: the food at home, the games children play in the street, the jokes everyone knows, the small rituals around festivals. These are the things a model can only get right if it truly carries your culture's knowledge.

## What makes a good riddle

- **It never names the answer.** Describe it through clues, wordplay, or the experience around it.
- **It has one fixed, short answer.** A reviewer should be able to say clearly whether an answer is right. If a few names are equally correct, list them in the answer, separated by a slash.
- **It needs your culture, not just your language.** Translating the riddle word for word should not be enough to solve it.
- **It is original.** Write it yourself. Don't copy riddles from books, websites, or other benchmarks, and don't translate riddles from another language.
- **It stays true over time.** Avoid answers that change, such as who currently holds an office.
- **It is fair.** Avoid content that targets or mocks a group of people. If a riddle touches on a painful part of history, say so in the comments.

Quality matters more than speed. A riddle that every member of your community would enjoy is worth more than one written only to fill a row.

## Topics

Your set covers ten topics with the same number of riddles in each. Eight topics are standard, and their names in the template must not change:

- Politics
- History
- Geography
- Arts
- Sports
- National Heritage
- Kid's World
- Everyday Life

The other two are **wildcard topics** that you define. Choose themes that are unique to your community's way of life, things that would not make sense as a generic global category. We want topics that show the distinct texture of your culture. Some ideas:

- **Public transit quirks:** matatu culture in Kenya.
- **Local climate and nature rhythms:** monsoon traditions in South Asia.
- **Social rituals:** sauna etiquette in Finland, or night market bargaining in Taiwan.
- **Childhood street games:** local rules that you won't find on Wikipedia.
- **Wedding and festival micro-traditions:** not the main event, but the specific things aunts and uncles do behind the scenes.

In the template, rename `Wildcard 1 [rename]` and `Wildcard 2 [rename]` to your two topics.

## Filling in the workbook

Each row of the `Riddles` sheet is one riddle:

| Column | What to write |
|---|---|
| `Number` | The riddle's number. Keep it the same if you correct the riddle later. |
| `Topic` | One of the eight standard topics or one of your two wildcard topics. |
| `Riddle (Original Language)` | The riddle in your language. Required. |
| `Riddle (English Translation)` | An English translation. Optional, but it helps reviewers. |
| `Hints (Optional)` | A nudge toward the answer without giving it away. |
| `Comments (Optional)` | Anything a reviewer should know: a dialect, a region, wordplay that does not survive translation, or a note on the answer. |
| `Reference Answer (Original)` | The fixed, short answer in your language. Required. |
| `Reference Answer (English)` | The answer in English. |

Two rows from the dataset show what a finished entry looks like. The first is from Tamil (India), so both language columns are filled in. The second is from English (Canada), where the wildcard topic is Food.

| Number | Topic | Riddle (Original Language) | Riddle (English Translation) | Hints (Optional) | Reference Answer (Original) | Reference Answer (English) |
|---|---|---|---|---|---|---|
| 74 | Everyday Life | கல்லூரிக்கு போகாமல் டிகிரி வாங்கினான், கோவில் ஊரில் பிறந்தான், டம்ளரிலிருந்து டவராவுக்கு குதித்து ஆடுவான். அவன் யார்? | He got a degree without going to college, he was born in a temple town, he jumps from tumbler(cup) to davara(saucer) and dances. Who is he? | He is a beverage. The "dance" is the pouring action. "Degree" is a play on words. | கும்பகோணம் டிகிரி காபி | Kumbakonam Degree Coffee |
| 23 | Food | You need 40 litres to make 1 litre. | | | Maple sap | Maple sap |

Neither riddle names its answer. The first rests on a pun and a pouring ritual that any Tamil reader knows. The second sounds like arithmetic but is about making maple syrup. No model in the benchmark solved it.

## Keep the workbook anonymous

Don't put your name, or anyone else's, in the workbook: not in a cell, a column, the file name, a comment, or the file's author properties. Submissions are public, and the dataset credits writers by community rather than by riddle. The submission check fails if it finds author details in the file, and `uv run python -m submissions clean` removes them.

If you would like to be credited as a writer, don't post your name in a public issue or pull request. A maintainer will ask you privately how you would like your name to appear.

## Before you submit

- Ask another native speaker to read your riddles. If they can't solve one, or find a second answer that also fits, reword it.
- Read every answer once more: is it short, fixed, and correct?
- Then follow [CONTRIBUTING.md](../CONTRIBUTING.md) to check your workbook and open a pull request.
