package org.example;

import org.example.clipBoard.GetFromClipBoard;
import java.util.concurrent.CountDownLatch;
import org.example.screen.ScreenSelector;
import org.example.tesseractOCR.FindTempFile;

import javax.swing.*;

public class Main {
     static void main(String[] args) throws Exception {
        CountDownLatch latch = new CountDownLatch(1); //Інструмент для синхронізації основний потік чекає поки лічильник не стане 0

        SwingUtilities.invokeLater(() -> {
            ScreenSelector selector = new ScreenSelector();
            selector.setOnDone(latch::countDown);
            selector.setVisible(true);
        }); //Запускається в асинхронном пуле і в кінці визивається latch який обнуляє лічільник до 0

        latch.await(); //Основний потік чекає поки користувач не завершить виділення

        GetFromClipBoard.Get();

        FindTempFile findTempFile = new FindTempFile();
        findTempFile.findImage();
    }
}
