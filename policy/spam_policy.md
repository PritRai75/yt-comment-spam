# Comment spam policy

**Version 1 (draft), 25 Nov 2026.** This is the single standard used for the label audit, the subtype labels and every classifier prompt. It is written for comments on music videos in this dataset and follows YouTube's spam, deceptive practices and scams policies; it is a working guideline for this project, not YouTube's internal rulebook.

## Definition

A comment is **spam** when its main purpose is to get something from the people reading it (a click, a subscription, a follow, a vote, money or personal details) for the benefit of the commenter or a third party, rather than to respond to the video or the conversation under it. Repetitive, untargeted comments posted in bulk are spam whatever they promote.

A comment is **ham** (not spam) when it is a reaction, opinion, question, joke or conversation about the video, the artist, the song or other comments, even if it is rude, off-colour, badly written or asks for likes on the comment itself.

Judge the comment as a viewer would see it (the cleaned text: visible link text, decoded characters). Judge the text only: not the author, the date or how often the author posted.

## Subtypes

When a comment fits more than one subtype, label the first that applies in this order: **scam or giveaway → link spam → self-promotion → bulk or off-topic**.

| Subtype | What it covers | Examples (invented) |
|---|---|---|
| **Scam or giveaway** | Free money, gift cards, prizes, giveaways, "earn $ from home", hacking tools, pyramid or referral schemes, requests for donations or for personal details | "Get free gift cards here, just enter your email"; "I made $500 today working from home"; "share this page to win a signed photo" |
| **Link spam** | A link (or an address written out, such as "site dot com") to another site, video, channel or page that is not the video being watched or its artist's official pages | "Best movies free http://example.com/watch"; "visit mysite dot com" |
| **Self-promotion** | Asking viewers to watch, subscribe to, follow or like the commenter's own (or a friend's) channel, music, page or account, with or without a link | "Check out my channel, I post covers"; "sub to me and I'll sub back"; "please like my pic on IG" |
| **Bulk or off-topic** | Copy-paste or templated text unrelated to the video, posted to many videos: chain messages, automatic share notices, repeated slogans | "Check out this video on YouTube:" with nothing else; the same paragraph pasted under unrelated videos |

## Edge cases

1. **A fan linking the video itself or the artist's official channel** ("the official video is here: youtu.be/…" pointing at this song): **ham**. Link spam needs a link that takes viewers somewhere else.
2. **"Subscribe to me" inside an otherwise on-topic comment** ("Love this song! Sub to my channel?"): **spam, self-promotion**. The request is for the commenter's benefit; the on-topic part is a wrapper.
3. **Asking for likes on the comment itself** ("Like if you're watching in 2015"; "1 like = 1 prayer"): **ham**. YouTube allows encouraging viewers to like, share or comment; this promotes nothing outside the thread.
4. **YouTube's automatic share text** ("Check out this video on YouTube:" or "…this playlist on YouTube:" with nothing else): **spam, bulk or off-topic**. It is auto-generated promotion posted in bulk and says nothing about the video. Because these are 122 comments (7 in dev, 115 in test, none in train), results are also reported with them removed.
5. **Abusive, hateful or offensive comments** about the artist or other viewers: **ham** for this policy. They break other YouTube policies, not the spam policy.
6. **Recommending another artist's song in a comparison** ("If you like this, Uptown Funk is better"): **ham** when it discusses music; **self-promotion** when the commenter is promoting their own or a friend's work ("my band covered this, check it out").
7. **Timestamps, hashtags and @/+ mentions** that link back to the video or to a person: not links for this policy. Judge the words.
8. **Votes and petitions for an outside contest** ("vote for my idol on this site"): **spam**, link spam if a link is given, otherwise self-promotion.
9. **Very short comments** (":)", "wow", emoji only): **ham** unless they contain a link or a request.

## How the policy is used

- The label audit reviews flagged items against this policy and records keep or correct, with a reason.
- Subtype labels on the 200-comment test sample use the precedence order above.
- The LLM prompts include this policy verbatim; comment text is passed as data, never as instructions.
- A change to this policy after either LLM prompt has been run on dev is a new version with a dated note; the test run uses the version committed before it.

## Sources

- YouTube Help, "Spam, deceptive practices & scams policies": https://support.google.com/youtube/answer/2801973
- YouTube Help, "Fake engagement policy": https://support.google.com/youtube/answer/3399767
