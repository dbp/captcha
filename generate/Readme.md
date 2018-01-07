This requires `maven` to build & run.

To build, run:

```
mvn package
```

To generate captcha images, create `out` directory and then run:

```
mvn exec:java -D exec.mainClass=captcha.App
```

This will put files named `challenge_randomnumber.png` in that directory. The
number is hard-coded in the source file. The randomnumber is so that multiple
captchas with the same challenge can be written.

To make the task easier, we then process the images (same thing will happen when
recognizing). This is done with imagemagick. The command, which turns it to
black and white and removes noise, is:

```
mogrify -extent 250x80 -morphology Smooth diamond:1 -threshold 50% -negate *
```

(if you use a shell, like `zsh`, that expands arguments on the commandline,
replace `*` with `\*` in the above). This should be run in the directory with
the generated images.
