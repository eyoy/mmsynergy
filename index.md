---
layout: default
---

Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# ABSTRACT

Although short-form dancing videos are rapidly emerging as a dominant format for both entertainment and marketing on social media platforms, there is a lack of systematic dance video quality measures that can guide influencers in crafting more impactful content and empower brands to predict multifaceted consumer engagement.

## Experiments
style>
  .video-container {
    display: flex;
    justify-content: center;
    gap: 20px; /* Space between videos */
    flex-wrap: wrap;
  }
  .video-container figure {
    text-align: center; /* Centers text below videos */
    width: 40%; /* Adjust width as needed */
  }
  .video-container video {
    width: 45%;
  }
</style>

<div class="video-container">
  <figure>
    <video controls>
      <source src="assets/6843971599771733254_H.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <figcaption>Video 1: Description of the first video</figcaption>
  </figure>

  <figure>
    <video controls>
      <source src="assets/6843971599771733254_H.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <figcaption>Video 2: Description of the second video</figcaption>
  </figure>
</div>


> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
