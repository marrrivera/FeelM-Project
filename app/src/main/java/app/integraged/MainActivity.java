package app.integraged;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.MediaPlayer;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import app.integraged.ml.Model;

public class MainActivity extends AppCompatActivity {

    // Declare UI elements
    TextView result, confidence;
    ImageView imageView;
    ImageButton picture;
    int imageSize = 224;
    MediaPlayer player;
    int currentSong = 0; // 0 for the happy song, 1 for the sad song

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI elements
        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.takepic);

        // Set a click listener for the "Take Picture" button
        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch the camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    // Request camera permission if we don't have it
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    // Method to classify an image using a pre-trained model
    public void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for the model
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Extract RGB values from the image and populate the ByteBuffer
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            // Load the image data into the input tensor
            inputFeature0.loadBuffer(byteBuffer);

            // Run model inference and get results
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Find the class with the highest confidence
            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            // Define class labels
            String[] classes = {"Sad", "Happy"};

            // Update UI with classification results
            result.setText(classes[maxPos]);

            // Play music based on the classification result
            playMusic(classes[maxPos]);

            String s = "";
            for (int i = 0; i < classes.length; i++) {
                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
            }
            confidence.setText(s);

            // Release model resources
            model.close();

        } catch (IOException e) {
            // Handle any exceptions
        }
    }

    private void playMusic(String emotion) {
        if (player != null) {
            player.release();
            player = null;
        }

        if (emotion.equals("Happy")) {
            player = MediaPlayer.create(this, R.raw.song_happy);
        } else if (emotion.equals("Sad")) {
            player = MediaPlayer.create(this, R.raw.song_sad);
        }

        if (player != null) {
            player.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                @Override
                public void onCompletion(MediaPlayer mp) {
                    stopMusic();
                }
            });
            player.start();
        }
    }

    public void pause(View v) {
        if (player != null) {
            player.pause();
        }
    }

    private void stopMusic() {
        if (player != null) {
            player.release();
            player = null;
        }
    }

    public void tryAgain(View v) {
        // Release the MediaPlayer instance
        stopMusic();

        // Create an intent to relaunch the main activity
        Intent intent = getIntent();
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_NO_ANIMATION);
        finish();
        startActivity(intent);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            // Resize the image to the expected input size and classify it
            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            // Then classify the image
            classifyImage(image);
            // Set visibility of text
            TextView confidencesText = findViewById(R.id.confidencesText);
            confidencesText.setVisibility(View.VISIBLE);
            TextView result = findViewById(R.id.result);
            result.setVisibility(View.VISIBLE);
            TextView confidence = findViewById(R.id.confidence);
            confidence.setVisibility(View.VISIBLE);
            TextView classified = findViewById(R.id.classified);
            classified.setVisibility(View.VISIBLE);
            ImageButton tryagain = findViewById(R.id.try_again);
            tryagain.setVisibility(View.VISIBLE);
            // Hide the picture icon
            picture.setVisibility(View.INVISIBLE);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    @Override
    protected void onStop() {
        super.onStop();
        stopMusic();
    }
}