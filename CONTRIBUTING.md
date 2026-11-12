# Add your community's riddles

AI models speak most major languages fluently, but they still miss the culture behind the words. In CulturalRiddles, even frontier models answer at most 22% of riddles correctly in some communities. Your riddles can show where models fall short for the people and places you know best.

**You don't need GitHub or any coding.** If you are a native speaker who grew up in a community, you can add it.

**What you get**

- Your name in the list of riddle writers for your community on the [Hugging Face dataset page](https://huggingface.co/datasets/CohereLabsCommunity/CulturalRiddles) and in this repository, if you'd like to be credited.
- Your culture in an open benchmark that researchers use to test how well AI models understand it.
- A place in a community of more than 160 native speakers from 62 communities who built it together.

## Communities we especially want

Any community that is not yet [in the dataset](docs/communities.md) is welcome. We especially want riddles from:

- **Oceania**
- **Africa**, beyond the communities already included
- **Central Asia**
- **Southeast Asia:** Laos, Cambodia, Myanmar, the Philippines
- **East Asia:** Mongolia, Korea

## How it works

1. **Tell us you're starting.** Email Vinod Anbalagan at vinodanbalagan@gmail.com, or open an [Add my community](https://github.com/Cohere-Labs-Community/CulturalRiddles/issues/new?template=add-community.yml) issue. We confirm your community and make sure no one else is already writing for it.
2. **Download the [template](data/submissions/submission_template.xlsx).** It's an Excel workbook with everything set up.
3. **Write 100 riddles**: 10 for each of 10 topics. You can write alone or with friends and family from your community.
4. **Ask another native speaker to check them.** If they can't solve a riddle, or find a second answer that also fits, reword it.
5. **Send us the workbook** (see below). We check it, review it, and add it to the dataset.

## Three ways to send your riddles

1. **Email it (no GitHub needed).** Send the workbook to Vinod Anbalagan at vinodanbalagan@gmail.com. We run the checks and submit it for you.
2. **Upload it on the GitHub website.** Fork this repository on github.com, choose **Add file** and then **Upload files**, and put your workbook in `data/submissions/`. GitHub offers to open a pull request, and the checks run on it automatically.
3. **Use git.** Run the checks yourself and open a pull request. See [Contributing with git](docs/contributing-with-git.md).

## Writing good riddles

A cultural riddle describes something without naming it. Someone who grew up in your community should recognise the answer at once. Someone who only knows the language, or searches the web, should struggle.

- **Never name the answer.** Describe it through clues, wordplay, or the experience around it.
- **Give one fixed, short answer.** If a few names are equally correct, list them separated by a slash.
- **Draw on your culture, not just your language.** The food at home, children's street games, jokes everyone knows, the small rituals around festivals.
- **Write it yourself.** Don't copy riddles from books, websites, or other benchmarks, and don't translate riddles from another language.
- **Keep it true over time.** Avoid answers that change, such as who currently holds an office.
- **Keep it fair.** Avoid content that targets or mocks a group of people.

**Two examples from the dataset:**

| Community | Topic | Riddle | Answer |
|---|---|---|---|
| Tamil (India) | Everyday Life | கல்லூரிக்கு போகாமல் டிகிரி வாங்கினான், கோவில் ஊரில் பிறந்தான், டம்ளரிலிருந்து டவராவுக்கு குதித்து ஆடுவான். அவன் யார்? *(He got a degree without going to college, he was born in a temple town, he jumps from tumbler to davara and dances. Who is he?)* | கும்பகோணம் டிகிரி காபி *(Kumbakonam Degree Coffee)* |
| English (Canada) | Food | You need 40 litres to make 1 litre. | Maple sap |

Neither names its answer. The first rests on a pun and a pouring ritual every Tamil reader knows. The second sounds like arithmetic but is about making maple syrup. No model in the benchmark solved it.

## The 10 topics

Eight topics are the same for every community: **Politics, History, Geography, Arts, Sports, National Heritage, Kid's World, and Everyday Life.**

The other two are **wildcard topics you choose**: themes unique to your community's way of life. For example:

- **Public transit quirks:** matatu culture in Kenya.
- **Climate and nature rhythms:** monsoon traditions in South Asia.
- **Social rituals:** sauna etiquette in Finland, or night market bargaining in Taiwan.
- **Childhood street games:** local rules you won't find on Wikipedia.
- **Wedding and festival micro-traditions:** the things aunts and uncles do behind the scenes.

In the template, rename `Wildcard 1 [rename]` and `Wildcard 2 [rename]` to your two topics.

## Filling in the template

Each row of the `Riddles` sheet is one riddle:

- **Riddle (Original Language):** the riddle in your language. Required.
- **Riddle (English Translation):** optional, but it helps reviewers.
- **Hints (Optional):** a nudge toward the answer without giving it away.
- **Comments (Optional):** anything a reviewer should know, such as a dialect, a region, or wordplay that doesn't survive translation.
- **Reference Answer (Original)** and **Reference Answer (English):** the short answer in your language, and in English.

On the `Community` sheet, fill in your language and country. We help you with the community key.

## Your name and credit

Keep the workbook anonymous: no names in the cells, the file name, or comments. The dataset credits writers by community, not by riddle. When you send your riddles, tell us privately how you'd like your name to appear, or that you'd rather not be named.

## Questions

- **My community is already in the dataset.** We're glad to hear about improvements. Open an issue or email us first, because a new submission replaces the whole set for that community.
- **Will my riddles be tested on AI models?** New communities join the dataset. The model results published with the paper cover the communities benchmarked for that release.
- **Do I need to be a researcher?** No. Riddles come from lived experience, and that is the expertise we need.
- **Anything else?** Email Vinod Anbalagan at vinodanbalagan@gmail.com.

Maintainers and git users: the checks, review, and import steps are in [Contributing with git](docs/contributing-with-git.md).
