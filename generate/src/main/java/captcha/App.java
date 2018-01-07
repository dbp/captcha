/**
   We generate 4 & 5 lowercase letter captchas, as that is what we observe in
   the wild.
 **/

package captcha;

import java.nio.file.Files;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import org.apache.wicket.extensions.markup.html.captcha.CaptchaImageResource;

class CaptchaBreaker extends CaptchaImageResource {
    public CaptchaBreaker(String string) { super(string); }
    public void toFile(String suffix) {
        byte[] bytes = this.render();
        String challenge = this.getChallengeId();
        try {
            Files.write(new File("out/" + challenge + "_" + suffix + ".png").toPath(), bytes);
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }
}

public class App
{
    public static void main( String[] args )
    {
        char[] letters = "abcdefghijklmnopqrstuvwxyz".toCharArray();

        // 1. Get a random number generator.
        Random rnd = new Random();

        // 2. Run for 25,000 times
        for (int i = 0; i < 25000; i++) {
            // NOTE(dbp 2018-01-05): FOR NOW ONLY 5s.
            // 3. If it's even, we generate a 5 letter captcha; odd we generate 4.
            int length = i % 2 == 0 ? 5 : 5;
            // 4. We want random lowercase letters. At current research, none
            // seem to be excluded.
            StringBuilder s = new StringBuilder(length);
            for (int j = 0; j < length; j++) {
                s.append(letters[rnd.nextInt(letters.length)]);
            }
            // 5. Now create the captcha with that challenge
            CaptchaBreaker captcha = new CaptchaBreaker(s.toString());
            // 6. And write out the file. Note that the same challenge will
            // render in different ways, so we want the file names not to just
            // reflect the challenge, hence this (hacky) way of adding some differentiation.
            captcha.toFile(Integer.toString(100000 + rnd.nextInt(100000)));
        }
    }
}
