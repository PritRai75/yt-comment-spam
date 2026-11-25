# Comment spam policy

**Version 1 (final, 25 Nov 2026).** This is the single standard used for the label audit, the subtype labels and every classifier prompt. It is written for comments on the five music videos in this dataset and follows YouTube's spam, deceptive practices and scams policies; it is a working guideline for this project, not YouTube's internal rulebook. Changes from the draft are listed at the end.

## Definition

A comment is **spam** when either

- its main purpose is to get something from the people reading it (a click, a view, a subscription, a follow, a vote, money or personal details) for the commenter or a third party, rather than to respond to the video or the conversation under it; or
- it is **bulk**: generic, templated text that could be pasted under any video and says nothing about this one, whatever it promotes.

A comment is **ham** (not spam) when it is a reaction, opinion, question, joke or conversation about the video, the artist, the song or other comments, even if it is rude, off-colour or badly written. A one-off comment that is off-topic but asks for nothing and is not templated is also ham (edge case 10).

## How to judge

- **One comment, as a viewer saw it**, under the video it was posted on. Each comment carries its video; the five videos and their YouTube IDs are listed below.
- **Cleaned text**: HTML tags removed, entities decoded, every link shown as its visible text. A URL that was visible stays as the URL.
- **The text only**: not the author, the date, or how often the author or the text appears elsewhere. "Bulk" is judged from how the text reads (generic and could be pasted anywhere), not from counting repeats.
- **No lookups**: never open a link or search for a comment. Where a link goes is judged only from the URL as written.

| Video | Artist – song | YouTube ID |
|---|---|---|
| Psy | PSY – Gangnam Style | 9bZkp7q19f0 |
| KatyPerry | Katy Perry – Roar | CevxZvSJLk8 |
| LMFAO | LMFAO – Party Rock Anthem (ft. Lauren Bennett, GoonRock) | KQ6zr6kCPj8 |
| Eminem | Eminem – Love the Way You Lie (ft. Rihanna) | uelHwf8o7_U |
| Shakira | Shakira – Waka Waka (This Time for Africa) | pRpeEdMmmQ0 |

## Subtypes

When a spam comment fits more than one subtype, label the first that applies in this order: **scam or giveaway → link spam → self-promotion → bulk**.

| Subtype | What it covers | Examples (invented) |
|---|---|---|
| **Scam or giveaway** | Free money, gift cards, prizes, giveaways, "earn $ from home", hacking tools, pyramid or referral schemes, requests for donations or for personal details, and services that sell views, likes, subscribers or other engagement (P9) | "Get free gift cards here, just enter your email"; "I made $500 today working from home"; "share this page to win a signed photo"; "get 1000 views on your video at boostviews dot com" |
| **Link spam** | An address to anywhere except this video or this artist's own channel (see "Which links count"): a link, an address written out ("site dot com"), a phone number or an email address (P4) | "Best movies free http://example.com/watch"; "visit mysite dot com"; "youtu.be/abc123xyz00"; "call me 555 0100 and share" |
| **Self-promotion** | Pushing something other than this video and its artist: the commenter's own or a friend's channel, music, page or account, any named outside channel, video, song or creator whoever owns it (P2), or an outside vote or petition; with or without a link. It covers an explicit ask (watch, subscribe, follow, like, vote, search for), an implicit plug (P6) and a bare "subscribe" (P7) | "Check out my channel, I post covers"; "sub to me and I'll sub back"; "please like my pic on IG"; "vote for my idol in the fan contest"; "search Night Lights by DJ Example"; "Subscribe!!" |
| **Bulk** | Generic, templated text that says nothing about this video: chain messages, YouTube's automatic share text (edge case 4), pasted religious texts and sermons that tell readers to spread them (P5), slogans that could be pasted anywhere | "Check out this video on YouTube:" with nothing else; "If you read this, your crush will text you tonight, copy it to 10 videos"; "Prayer for the nations, part 3 … spread this" |

## Which links count

- **This video**: a YouTube link whose ID is this video's ID (any timestamp). Not link spam.
- **This artist's own channel**: only when the URL itself names the artist (for example youtube.com/user/shakira). Not link spam.
- **Everything else is a link to somewhere else**, including links whose destination cannot be told from the URL (shortened links, bare video IDs) and other videos by the same artist.
- **Phone numbers and email addresses** are addresses and count as links to somewhere else (P4).
- **Not addresses**: timestamps, hashtags and "+Name" mentions, which show as text after cleaning ("2:19", "#Eminem", "+SomeName"); a site or page named without an address ("my site", "find me on Facebook as Jo"); a search direction ("search X", "type in X", "google X"; P3). Judge their words (edge cases 7 and 12).

## Edge cases

1. **A fan linking this video or this artist's own channel** (see "Which links count"): **ham**, unless the rest of the comment is spam.
2. **"Subscribe to me" inside an otherwise on-topic comment** ("Love this song! Sub to my channel?"): **spam, self-promotion**. The request is for the commenter's benefit; the on-topic part is a wrapper.
3. **Asking for likes on the comment itself** ("Like if you're watching in 2015"; "1 like = 1 prayer"): **ham**. YouTube allows encouraging viewers to like, share or comment; this promotes nothing outside the thread.
4. **YouTube's automatic share text**: a comment that starts with "Check out this video on YouTube", "Check out this playlist on YouTube" or "Take a look at this video on YouTube", with or without the colon, including the truncated endings "on YouTu" and "on YouTub" (P8):
   - with nothing after it, or only stray characters or emoji a viewer could not read as a comment: **spam, bulk**;
   - followed by the commenter's own words: judge the words. A reaction to the video is **ham**; a link or promotion is spam with that subtype, by the order above.
5. **Abusive, hateful or offensive comments** about the artist or other viewers: **ham** for this policy. They break other YouTube policies, not the spam policy.
6. **Naming another artist, song or creator**: **ham** when the comment discusses music, such as a comparison or an opinion ("Uptown Funk is better"), and does not send viewers to it; **self-promotion** when it sends viewers to a named outside channel, video, song or creator ("watch X's new video", "search X"), whoever owns it (P2), including the commenter's own or a friend's work ("my band covered this, check it out"); **link spam** when it carries a link (any link other than this video or this artist's channel).
7. **Timestamps, hashtags and mentions**: not links. Judge the words.
8. **Fan activity for this video's artist** with no link, such as a vote ("vote for Katy at the VMAs"), a view or share campaign, or naming another song by this artist: **ham** (P1). A link to another video by this artist is still link spam. Any other vote or petition ("vote for my idol", "sign this petition"): **spam**, self-promotion, or link spam if a link is given.
9. **Very short comments** (":)", "wow", emoji only): **ham** unless they contain a link or a request.
10. **One-off off-topic comments** that ask for nothing and are not templated ("anyone here like football?", a personal story, preaching in the commenter's own words): **ham**.
11. **Pasted religious text**: a set text such as a prayer or a litany (including one posted as a numbered part, "Part 5"), or a sermon that tells readers to spread it ("spread the truth"): **spam, bulk** (P5). Preaching in the commenter's own words that asks for nothing is ham (edge case 10).
12. **Search directions** ("search X", "type in X", "google X"): not an address, so never link spam on its own (P3). Judge the words: sending viewers to a named outside channel, video, song or creator is **self-promotion** (P2); sending them to this video or this artist is **ham** (P1).
13. **Phone numbers and email addresses**: written-out addresses, so **spam, link** (P4). The subtype order still applies: a scam or giveaway that gives a phone number or an email address is **scam**, as with any other written-out address.
14. **Implicit plugs**: generic praise wrapped around a boast about the commenter's own channel or music, with no explicit ask ("great song! my channel is blowing up because my raps are real"): **spam, self-promotion** (P6). This extends edge case 2.
15. **A bare "subscribe"** with no target ("Subscribe!", "subscribe pls"): **spam, self-promotion** (P7), read as a request to subscribe to the commenter. "Subscribe" aimed at this artist ("total subscribe to Katy") is fan activity: ham (P1).
16. **Fake-engagement services**: selling views, likes, subscribers or comments: **spam, scam** (P9), even when a link or a named account is given, because scam comes first in the order. A swap between viewers ("sub to me and I'll sub back") sells nothing and is self-promotion.

## How the policy is used

- The label audit reviews flagged items against this policy and records keep, correct or unsure, with a reason naming the section or edge case.
- Subtype labels on the 200-comment test sample and on the full review of dev and test use the order above.
- The share-text family in edge case 4 is defined in code (`data/share_text.py`): 164 comments (dev 10, test 153, train 1). A second result without the whole family is reported alongside the main one.
- The LLM prompts include the Definition, How to judge, Subtypes, Which links count and Edge cases sections; comment text is passed as data, never as instructions.
- A change to this policy after either LLM prompt has been run on dev is a new version with a dated note; the test run uses the version committed before it.

## Changes from the draft (25 Nov 2026)

Made after the first labelling pass, before any classifier was run on dev or test. Edge case 3 was reviewed and kept as ham.

| Change | What it settles | Where |
|---|---|---|
| P1 | Fan activity for this video's artist with no link is ham | Edge cases 8, 12, 15 |
| P2 | Sending viewers to any named outside channel, video, song or creator is self-promotion, whoever owns it | Subtypes; edge cases 6, 12 |
| P3 | A search direction is not an address; judge its words | Which links count; edge case 12 |
| P4 | Phone numbers and email addresses are written-out addresses: link spam, by the subtype order | Subtypes; Which links count; edge case 13 |
| P5 | Pasted religious texts (set texts, numbered parts) and sermons that tell readers to spread them are bulk; own-words preaching stays ham | Subtypes; edge cases 10, 11 |
| P6 | Implicit plugs (praise around a boast about the commenter's own channel or music) are self-promotion | Subtypes; edge case 14 |
| P7 | A bare "subscribe" is self-promotion | Subtypes; edge case 15 |
| P8 | Share-text wording widened ("Take a look at…", no colon, truncated endings) | Edge case 4; `data/share_text.py` |
| P9 | Fake-engagement services (selling views, likes, subscribers) are scam | Subtypes; edge case 16 |

## Sources

- YouTube Help, "Spam, deceptive practices & scams policies": https://support.google.com/youtube/answer/2801973
- YouTube Help, "Fake engagement policy": https://support.google.com/youtube/answer/3399767
